import sys
import struct
import serial
from time import sleep
import numpy as np


class CyberGloveException(serial.SerialException):
    """Class for CyberGlove specific errors"""


class CyberGlove(serial.Serial):

    MAX_BAUD = 115200   # Maximum data transfer rate in bits per second
    MAX_WAIT = 57600    # Maximum wait steps between samples, equivalent to min sample rate of 2Hz

    SENSOR_DIGITAL_T = 0.00025     # Time for each sensor's data to be digitize (in seconds)
    DIGITAL_FILTER_T = 0.0001256   # Time for each sensor's data to pass through the digital filter (in seconds)
    
    def __init__(self, port, **kwargs):
        """
        Initialize serial port level communication with the CyberGlove

        :param port: name of the port to access
        :param kwargs: for a list of all other key word arguments, see serial.Serial
            Several parameters have special default values set here:
            timeout: default to 1 second
        """

        if 'timeout' not in kwargs.keys():
            kwargs['timeout'] = 1
        if 'baudrate' not in kwargs.keys():
            kwargs['baudrate'] = 115200   # Data transfer rate in bits_per_second

        super(CyberGlove, self).__init__(port, **kwargs)

        self.num_sensors = self.get_sensors()
        if self.num_sensors == 0:
            raise CyberGloveException("Failed to get the number of active sensors! Try restarting the glove")

    def __str__(self):
        return f"CyberGlove on {self.name}"

    def __del__(self):
        """Ensure a stop command is sent before the glove interface is deleted"""
        self.stop()
        super(CyberGlove, self).__del__()

    @property
    def max_rate(self):
        return self.est_max_rate()

    def close(self):
        """Ensure the glove stops streaming data before closing the socket/interface"""
        self.stop()
        super(CyberGlove, self).close()

    def stop(self):
        """Send a signal to stop execution of the current command and return to a resting state"""
        return self.write(b'\x03')

    def newline(self):
        """Send a carriage return an EOL signal"""
        return self.write(b'\r\n')

    def clear(self):
        """Harshly terminate all commands and clear the read buffer"""
        self.newline()
        self.stop()
        # Keep reading until the buffer is guaranteed empty
        sleep(0.25)  # Allow some time for data to be written to the buffer
        read = self.read_all()
        while read:
            sleep(0.05)  # Allow some time for new data to arrive
            print(f"Cleared {read}")
            read = self.read_all()

    def stream_sensors(self):
        """
        Return a generator that allows for iteration over the live data stream from the glove

        To ensure no unexpected data is returned, make sure to close the generator before issuing any other commands.
            Do this simply by calling .close() on the generator object
        The generator will close itself if it encounters data that does not seem to belong to a sensor data stream, but
        this may still result some stream data being dumped in the command response.
        """
        self.write(b'S')     # Send the command for the CyberGlove to begin streaming data (don't read)

        try:
            while True:
                # Read through the data for the current state which is terminated by a null character
                buffer_data = self.read_until(b'\x00')

                # As long as the glove is streaming data, each sensor state set will begin with an S (ascii 83)
                if buffer_data[0] != 83:
                    raise GeneratorExit('Received data that was not part of the sensor stream.')
                this_state = self.extract_state(buffer_data)
                yield this_state

        # Ensure that streaming is stopped and buffer is cleared when leaving the generator
        finally:
            self.stop()
            self.read_all()

    def get_status(self):
        """
        Return the status of the glove as a 2-tuple of the ready flag and a message

        :return: 2-tuple. The first element is a boolean value for whether the glove is ready to operate or not, and the
        second value is a message giving detail about the state of the glove.
        :raises: If a unexpected status is received, will raise a CyberGloveException
        """
        response = self.send_command('?G')
        status_int = self.extract_value(response, 2)

        if status_int == 0:
            status = (False, "Glove not plugged in!")
        elif status_int == 1:
            status = (False, "Glove plugged in but not initialized!")
        elif status_int == 2:
            # This should not be possible to reach since glove can't initialize unless it's plugged in
            status = (False, "Impossible! Glove not plugged in, but initialized!")
        elif status_int == 3:
            status = (True, "Glove ready!")
        else:
            raise CyberGloveException("Unknown glove status received!")

        return status

    def get_sensors(self):
        """Return the number of sensors available for data retrieval"""
        response = self.send_command('?N')
        # Response will be b'?N<#sensors>\x00', so get only the one byte
        num = self.extract_value(response, 2)
        return num

    def get_state(self):
        """Get the current state of all the glove sensors"""
        response = self.send_command('G')
        sensor_states = self.extract_state(response)
        return sensor_states

    def get_filter_status(self):
        """Determine if the digital filter is turned on or off"""
        response = self.send_command('?F')
        filter_status = self.extract_value(response, 2)
        return filter_status

    def get_sampling_rate(self):
        """
        Return the current sampling rate (frequency) in Hz

        If the sampling rate is unset, then the Glove is returning data as fast as it can. The returned value for rate
        will be INF, since inter-sample interval (ait_counts) is 0.
        In this case the effective sampling rate will be close to the max sampling rate. See ext_max_rate()
        """
        response = self.send_command('?T')

        # Data is returned as 16-bit big-endian values, so we must decode manually
        data_range = response[2:6]
        period = struct.unpack('>HH', data_range)

        wait_counts = period[0] * period[1]  # Number of counts to wait between samples
        try:
            rate = self.MAX_BAUD / wait_counts
        except ZeroDivisionError:
            # Suppress the divide by zero warning and explicitly set an infinite rate
            rate = float('Inf')
        return rate

    def est_max_rate(self):
        """
        The max sampling rate is driven by the number of sensors and the baud rate

        The rate is based on the underlying hardware and can be estimated by:

            1/f = t_c + t_d

        Where t_c is the time to transfer the data:
            t_c = (10 * N )/ R          where N is the number of sensors and R is the baud rate
        And the time to digitize the data is
            t_d = N * d                 where is the time needed to digitize the data for one sensor. When the digital
                                    filter is disabled this is ~0.25 ms otherwise ~0.375 ms

        :return: Estimated max firing rate for the current settings in Hz
        """
        if self.get_filter_status():
            digital_delay = self.SENSOR_DIGITAL_T + self.DIGITAL_FILTER_T
        else:
            digital_delay = self.SENSOR_DIGITAL_T

        transfer_time = (10 * self.num_sensors) / self.baudrate
        digital_time = self.num_sensors * digital_delay

        period = transfer_time + digital_time
        rate = 1 / period
        return rate

    def set_filter_status(self, new_status=1):
        """Set the on/off status of the digital filter"""
        return self.set_param_flag('f', new_status)

    def set_sampling_rate(self, rate):
        """Set the sampling rate to as close to the given rate (Hz) as possible"""
        wait_counts = round(self.MAX_BAUD / rate)
        print(wait_counts)
        if wait_counts > self.MAX_WAIT:
            wait_counts = self.MAX_WAIT
        period = struct.pack('>HH', wait_counts, 1)

        # Since we're dealing with non-standard (big endian, 16-bit) values, we have to handle encoding manually
        command = b''.join([b'T', period])
        response = self.send_command(command, encoded=True)
        return response

    def set_param_flag(self, flag, new_value):
        """
        Send a command to set the given parameter flag to the new value

        Setting parameter flags tends to run into issues, handle these here

        :param flag: single character, name of parameter flag to set
        :param new_value: 0 or 1, the new value of the flag
        """
        command = f'{flag.lower()}{new_value}'.encode(encoding='utf-8')
        command = b''.join([command, b'\r\n'])
        response = self.send_command(command, encoded=True)
        self.clear()
        return response

    def get_hardware_calibration(self):
        """
        Get the hardware calibration values

        These are the factory hardware calibration values defined to prevent sensor saturation. These should not be
        changed unless there seems to be a hardware issue with the sensors

        :return: 2-tuple of numpy arrays: the first element is all the gains, the second element is all the offsets
        """
        response = self.send_command('?C')
        second_line = self.readline()

        # Concatenate the two lines and remove any spaces between values
        all_bytes = b''.join([response, second_line])
        split = all_bytes.split(b' ')
        all_bytes = b''.join(split)

        all_values = self.extract_values(all_bytes, 2, 3*self.num_sensors)

        # The extracted array is all the sensor indices, followed by all the offsets and gains, so split it up
        indices = all_values[:self.num_sensors]
        offsets = all_values[self.num_sensors:(2*self.num_sensors)]
        gains = all_values[(2*self.num_sensors):]

        print("getting hardware calibration.")
        return gains, offsets

    def set_sensor_calibration(self, index, gain=None, offset=None):
        """
        Set the hardware calibration for a single sensor

        These are the factory hardware calibration values defined to prevent sensor saturation. These should not be
        changed unless there seems to be a hardware issue with the sensors

        :param index: integer index of the
        :param gain: optional integer value to set the hardware amplifier gain to. If omitted gain is left unchanged
        :param offset: optional integer value to set the hardware offset to. If omitted offset is left unchanged
        :return: None
        """
        if offset is not None:
            offset_command = f"CO {index} {offset}"
            self.send_command(offset_command)

        if gain is not None:
            gain_command = f"CG {index} {gain}"
            self.send_command(gain_command)

    def send_command(self, command, encoded=False):
        """
        Send the given command to the CyberGlove and return the single line response

        Note: Do not use this function with any streaming type commands, as it will:
            - only return the first line and let data pile up in the buffer (if stream contains EndOfLine characters)
            - freeze the program as the buffer waits for an EOL (if the streamed data does not contain any EOL)

        :param command: string version of the command, to be utf-8 encoded and sent to the glove
            see SerialCommands.pdf for a list and explanation of all available commands.
        :param encoded: optional parameter. If set to True, will not attempt to byte encode the passed command
        :return the response of the glove returned in byte form
        """
        if encoded:
            byte_command = command
        else:
            byte_command = command.encode(encoding='utf-8')
        self.write(byte_command)
        response = self.readline()

        # If the previously issued command was human readable, there may be an extra null char at the line start
        if response and response[0] == 0:
            response = response[1:]

        # If the response (after the issued command) begins with an 'e', then we've received an error code
        error_check = response[len(command):len(command) + 2]
        if b'e' in error_check:
            self.handle_error(command, response)

        return response

    def send_query(self, query, encoded=False):
        """
        Send a query to the glove and attempt to return a human readable response

        :param query: string for of a query, see SerialCommands.pdf. Since this function attempts to return the response
            in a human-readable format, it is recommended to use the lower-case versions of the commands
        :param encoded: optional parameter. If set to True, will not attempt to byte encode the passed command
        :return: utf-8 decoded response from the glove
        """
        response = self.send_command(query, encoded=encoded)
        answer = response.decode(encoding='utf-8')
        return answer

    def extract_state(self, glove_response):
        """
        Extract the integer state values of all the sensors form a byte-array format of the response

        :param glove_response: the response of the glove to either a 'G' or 'S' query. Data must have been returned in
            byte format, so the upper case version of the command should be used
        :return: numpy array of integers: the sensor values for all sensors on the glove
        """
        sensor_values = self.extract_values(glove_response, 1, self.num_sensors)
        return sensor_values

    @staticmethod
    def extract_value(glove_response, value_index):
        """
        Extract a single integer value from a list of bytes at the specified index

        :param glove_response: the response of the glove, as bytes or bytearray
        :param value_index: integer index of the value to extract in the byte array
        :return: integer value
        """
        byte = glove_response[value_index:value_index+1]
        value = int.from_bytes(byte, byteorder=sys.byteorder)
        return value

    @staticmethod
    def extract_values(glove_response, prefix_length, values_to_extract):
        """
        Extract an array of values fromm the given response

        :param glove_response: the response of the glove, as bytes or bytearray
        :param prefix_length: the number of extraneous bytes at the front of the array to skip
        :param values_to_extract: the number of values to extract from the array
        :return: numpy ndarray of integer values
        """
        last_index = prefix_length + values_to_extract
        byte_values = glove_response[prefix_length:last_index]
        values = np.frombuffer(byte_values, dtype=np.uint8)
        return values

    @staticmethod
    def handle_error(command, glove_response):
        """
        Raise a CyberGloveException with a message corresponding to the type of error that occurred

        :param command: The command that was issued to the glove
        :param glove_response: The response (in bytes) that was received from the glove
        """

        # If the command was sent in lower case, the response will be in ascii and include a space
        if command.islower():
            error_index = len(command) + 2
        else:
            error_index = len(command) + 1

        # Error codes are 2 characters, where the second character is unique to the type of error that occurred
        error_char = glove_response.decode(encoding='utf-8')[error_index]
        if error_char == '?':
            raise CyberGloveException('Unknown command!')
        elif error_char == 's':
            raise CyberGloveException('Sampling rate is set too high!')
        elif error_char == 'n':
            raise CyberGloveException('Too many numbers entered!')
        elif error_char == 'y':
            raise CyberGloveException('Sync input rate is too fast!')
        elif error_char == 'g':
            raise CyberGloveException('Glove not plugged in!')
        else:
            raise CyberGloveException(f'Error with unknown error code: {error_char}')




