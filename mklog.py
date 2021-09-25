import logging
import os


def get_logger(outputdir,logname):
    logger = logging.getLogger(__name__)

    logformat=logging.Formatter("[%(asctime)s|%(levelname)s]:%(message)s")
    filename=f'{outputdir}/{logname}.log'
    if os.path.isfile(filename):
    	os.remove(filename)

    file_handler=logging.FileHandler(filename)
    stream_handler=logging.StreamHandler()


    # logging.basicConfig(filename=f'{outputdir}/{logname}.log', level=logging.DEBUG,format=logformat)
    logger.setLevel(logging.DEBUG)

    stream_handler.setFormatter(logformat)
    file_handler.setFormatter(logformat)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


    return logger



