import zmq
import numpy as np
from glove import CyberGlove, CyberGloveException


def list_ports():
    """
    Get a simple list of al the available ports.

    For greater functionality, especially optional filtering by a regex expression, use:
    python -m serial.tools.list_ports
    """
    from serial.tools.list_ports import comports
    ports = comports(False)
    for info in ports:
        print(info)
    return


def open_glove(port, **kwargs):
    """
    Create and return a CyberGlove object on the given port
    This function also checks the status of the CyberGlove and will raise CyberGloveException if the glove is not ready

    :param port: the name of the serial port the CyberGlove is connected to. Usually this will be something like 'COM6'
    see glove.CyberGlove for list of available key word arguments
    :return: instance of CyberGlove, a wrapper around serial.Serial specifically for communicating with a CyberGlove
    """
    glove = CyberGlove(port, **kwargs)
    #print(f"Got connection to: {glove}")

    ready, message = glove.get_status()
    if not ready:
        raise CyberGloveException(message)

    return glove


def connect_state_stream(address):
    """
    Create a socket that will listen to published events and try to interpret them as a glove sensor state

    This function opens a zmq subscriber socket on the given address, and listens for any and all incoming messages.
    Whenever it receives a message (expected to be in JSON format), it will try to interpret the message as a numpy
    array of glove sensor states.

    Implementation wise, this function is a generator, so the returned object behaves like an iterable, where each next
    item is the next state received in the stream. Simply throw it in a for loop and iterate through the sensor states,
    but be sure to close the stream when you are done.

    :param address: Address (including transport protocol) to listen on
    :return: Generator object of the senor state stream
    """

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    get_all = ''    # Empty filter => receive all messages

    subscriber.connect(address)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, get_all)

    print('Subscribed. Listening...')

    try:
        while True:
            # Wait for the
            data = subscriber.recv_json()
            state = np.array(data)
            yield state

    # Ensure that streaming is stopped and buffer is cleared when leaving the generator
    finally:
        subscriber.close()
        context.term()




