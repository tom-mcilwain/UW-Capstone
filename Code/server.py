import os
import zmq
from connect import open_glove


STREAM_LOCATION = os.environ.get('GLOVE_STREAM_LOCATION', "127.0.0.1:5556")
REPLY_LOCATION = os.environ.get('GLOVE_REPLY_LOCATION', "127.0.0.1:5557")

CONNECTION_PROTOCOL = os.environ.get('COMMUNICATION_PROTOCOL', "tcp")


def make_context():
    """Create and return a ZeroMQ (zmq) server context that can be used to connect to create sockets"""
    context = zmq.Context()
    return context


def publish_glove_state(glove, sampling_rate=None, calibration=None):
    """
    Open a CyberGlove object and create a live-stream of the data that can be captured in any other process

    A CyberGlove object is opened on the given port, and a data stream is initialized (see the stream_sensors() method
    on the CyberGlove object). Each state of the glove is then published on a ZMQ publisher socket, so that it can
    be subscribed to and receive the live feed from the CyberGlove.

    The state is converted to a simple python list before being published.

    :param glove: an open and ready CyberGlove object. Will not be available until publishing is finished
    :param sampling_rate: optional. If given will set the CyberGlove to sample at this rate
    :param calibration: optional. If given, this calibration will be applied to each state before it is published
    :return: None.
    """

    # Create the context and publisher
    context = make_context()
    publisher = context.socket(zmq.PUB)
    address = f"{CONNECTION_PROTOCOL}://{STREAM_LOCATION}"
    publisher.bind(address)

    # Begin streaming data at the desired rate
    if sampling_rate is not None:
        glove.set_sampling_rate(sampling_rate)
    sensor_stream = glove.stream_sensors()

    print(f"Opening glove state stream on '{address}'...")

    try:
        for state in sensor_stream:

            # Apply the given calibration if it has been passed
            if calibration is not None:
                state = calibration(state)

            # Convert the state (numpy array) to a simple python list for simplicity
            simple_state = state.tolist()
            publisher.send_json(simple_state)

    # Ensure that the stream, socket, and context are always closed on error or exit
    finally:
        print("Closing glove state stream...")
        sensor_stream.close()
        publisher.close()
        context.term()


def make_glove_state_reply(glove, calibration=None):
    """
    Create a ZMQ socket that will wait until any message is received and reply with the current state of the glove

    A CyberGlove object is opened on the given port and begins to wait. Whenever any message except a termination signal
    is received, the socket will get and reply with the current state of the glove sensors. When a termination signal is
    received, it will trigger a KeyboardInterrupt error and will cause the socket to shut down.

    :param glove: an open and ready CyberGlove object. Will not be available until the reply server is closed
    :param calibration: optional. If given, this calibration will be applied to the state before it is sent in reply
    :return: None
    """

    # Create the context and publisher
    context = make_context()
    replier = context.socket(zmq.REP)
    replier.bind(f"{CONNECTION_PROTOCOL}://{REPLY_LOCATION}")

    print("Connection ready. Waiting...")

    try:
        while True:
            # Wait here until any message is received
            message = replier.recv()
            print(f'Received {message}. Replying with glove state...')

            # Get the glove state and apply a calibration if it was passed
            state = glove.get_state()
            if calibration is not None:
                state = calibration(state)

            # Send the transformed and simplified (numpy array -> list) glove state back to the requester
            simple_state = state.tolist()
            replier.send_json(simple_state)

    except KeyboardInterrupt:
        print("Termination received. Closing...")

    # No matter how we exit listening, make sure the socket and context are correctly closed
    finally:
        replier.close()
        context.term()
        print("Clean Shutdown complete")



