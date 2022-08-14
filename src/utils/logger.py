import logging

# suppress all the urllib3 noise about XML message format
logging.getLogger('urllib3').setLevel(logging.ERROR)

# setup graylog and console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('polymer_analytics_api')

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)
