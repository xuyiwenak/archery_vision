from datetime import datetime
import logging

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()


def log_process(f):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        processed = 0
        video = args[1]
        for frame, result in f(*args, **kwargs):
            yield frame, result
            # 进度日志
            processed += 1
            if processed % 30 == 0:  # 每30帧输出一次进度
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_log = processed / elapsed
                remain = (video.total_frames - processed) / fps_log if fps_log > 0 else 0
                logger.info(
                    f"⏳ 进度: {processed}/{video.total_frames} "
                    f"({processed/video.total_frames:.0%}) | "
                    f"耗时: {elapsed:.1f}s | "
                    f"剩余: {remain:.1f}s"
                )

    return wrapper