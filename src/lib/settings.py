import logging
import time


def create_logger(name: str) -> logging.Logger:
    """loggerの設定を行う関数
    Args:
        name (str): loggerの名前
    Returns:
        logger (logging.Logger): logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    format = "%(levelname)-9s  %(asctime)s [%(filename)s:%(lineno)d] %(message)s"

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(format))

    logger.addHandler(stream_handler)
    return logger


class Timer:

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, ex_type, ex_value, trace):
        self.t_end = time.time()
        logger.debug(f"{self.name} : {self.t_end - self.t_start} sec")


def time_dec(func):
    def wrapper(*args, **kwargs):
        print(f"start {func.__name__}")
        with Timer(func.__name__):
            func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    # loggerの生成
    logger = create_logger("simplex")

    # 関数自体の速度を測るなら@time_decをつける
    # ある処理の速度が知りたいなら、その処理の中身にwith Timer(name) → nameの中身は任意
    # 例えば、with Timer("base")とすると、任意でつけたbaseという名前の処理時間が出力される

    # 今回の例では、sample関数全体の処理を測りたいので、@time_decをつけている
    # また、関数内部の処理時間を知るために、with Timer("base")をつけている

    @time_dec
    def sample(logger: logging.Logger):
        with Timer("base"):
            logger.debug("start")
            time.sleep(1)
            logger.debug("end")
        sample(logger)
