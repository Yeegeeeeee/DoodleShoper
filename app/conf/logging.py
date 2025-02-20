import os
import yaml
import logging
import logging.config


class SensitiveInfoFilter(logging.Filter):
    # Mask sensitive information in logs
    def __init__(self, sensitive_words):
        super().__init__()
        self.sensitive_words = sensitive_words

    def filter(self, record):
        # This breaks with openai responses so commented it
        # # Check sensitive information
        # msg = record.getMessage()
        # for word in self.sensitive_words:
        #     msg = msg.replace(word, "*" * len(word))
        # record.msg = msg
        return True


def replace_env_for_config(log_conf: dict) -> None:
    for k, v in log_conf.items():
        if isinstance(v, dict):
            replace_env_for_config(v)
        elif isinstance(v, str) and v[0] == '$':
            log_conf[k] = os.environ.get(v[1:])
    return None


def create_log_config(log_path: str) -> dict:
    with open(log_path, 'r') as f:
        log_config = yaml.load(f, Loader=yaml.CLoader)
        replace_env_for_config(log_config)
    return log_config


def setup_logging():
    log_config = create_log_config('conf/logging.yaml')
    logging.config.dictConfig(log_config)