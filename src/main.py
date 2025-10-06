from logger import get_logger

def main():
    logger = get_logger("example_service", level="DEBUG")

    logger.info("Service starting up")
    logger.debug("This is debug information")
    logger.warning("This is a warning")

    try:
        # Simulate some work
        result = 10 / 2
        logger.info(f"Calculation result: {result}")
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)

    logger.info("Service shutting down")

if __name__ == "__main__":
    main()
