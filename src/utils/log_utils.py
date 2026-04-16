import logging
import os

def setup_logger(log_dir="logs", log_file="train.log"):
    """ 设置日志记录 """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

def log_info(message):
    """ 记录日志信息 """
    logging.info(message)

def log_error(message):
    """ 记录错误信息 """
    logging.error(message)

def log_metrics(epoch, rec_loss, interest_loss, hr, ndcg, mrr, category_acc):
    """
    记录多任务学习指标
    - epoch: 当前训练轮数
    - rec_loss: 推荐任务的损失
    - interest_loss: 兴趣预测任务的损失
    - hr: 命中率 (HR@K)
    - ndcg: 归一化折扣累积增益 (NDCG@K)
    - mrr: 均值倒数排名 (MRR@K)
    - category_acc: 兴趣类别预测准确率
    """
    message = (f"Epoch {epoch}: "
               f"Rec Loss={rec_loss:.4f}, Interest Loss={interest_loss:.4f}, "
               f"HR={hr:.4f}, NDCG={ndcg:.4f}, MRR={mrr:.4f}, "
               f"Category ACC={category_acc:.4f}")
    logging.info(message)
