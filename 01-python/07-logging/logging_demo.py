import logging 

## Configuring the logging

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.FileHandler("app1.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ArthmaticApp")

def add(a,b):
    result = a + b
    logger.debug(f"Adding {a} + {b}, result: {result}") 
    return result 

def subtract(a,b):
    result = a - b
    logger.debug(f"Subtracting {a} - {b}, result: {result}") 
    return result

def multiply(a,b):
    result = a * b
    logger.debug(f"Multiplying {a} * {b}, result: {result}") 
    return result

def divide(a,b):
    if b == 0 or a == 0:
        logger.error("Division by zero attempted")
        return None
    result = a / b
    logger.debug(f"Dividing {a} / {b}, result: {result}") 
    return result


add(10,15)
subtract(20,5)
multiply(5,5)
divide(10,0)