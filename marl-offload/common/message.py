import zmq
from common.config import TRAINER_IP, ZMQ_PORT_ROLLOUT, ZMQ_PORT_CMD

def trainer_pull_ctx():
    ctx = zmq.Context.instance()
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://0.0.0.0:{ZMQ_PORT_ROLLOUT}")
    pub  = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://0.0.0.0:{ZMQ_PORT_CMD}")
    return ctx, pull, pub

def agent_push_ctx():
    ctx = zmq.Context.instance()
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://{TRAINER_IP}:{ZMQ_PORT_ROLLOUT}")
    sub  = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{TRAINER_IP}:{ZMQ_PORT_CMD}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe all
    return ctx, push, sub
