import os
import datetime
import logging
import torch

class Logger(object):
    def __init__(self,writetofile=False):
        super(Logger, self).__init__()
        os.chdir(os.path.dirname(__file__))
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y-%m-%d-%H:%M:%S')
        
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)   # Log等级总开关
        if writetofile:
            # 第二步，创建一个handler，用于写入日志文件
            logfile = './log/' + time_str.replace(':', '-') + '.txt'
            createfile = open(logfile, 'w')
            createfile.close()
            fh1 = logging.FileHandler(logfile, mode='w',encoding='utf-8')
            fh1.setLevel(logging.DEBUG)  # 用于写到file的等级开关

            logfile = './log/' + time_str.replace(':', '-') + '_statistic.txt'
            createfile = open(logfile, 'w')
            createfile.close()
            fh2 = logging.FileHandler(logfile, mode='w',encoding='utf-8')
            fh2.setLevel(logging.INFO)  # 用于写到file的等级开关
        
        # 第三步，再创建一个handler,用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)    # 输出到console的log等级的开关
        
        # 第四步，定义handler的输出格式
        formatter1 = logging.Formatter('%(asctime)s - %(message)s')
        formatter2 = logging.Formatter('%(message)s')
        ch.setFormatter(formatter1)
        if writetofile:
            fh1.setFormatter(formatter1)
            fh2.setFormatter(formatter2)
        
        # 第五步，将logger添加到handler里面
        self.logger.addHandler(ch)
        if writetofile:
            self.logger.addHandler(fh1)
            self.logger.addHandler(fh2)
        
    
    def debug(self, *messages):
        message = ''
        for m in messages:
            message += str(m)
        self.logger.debug(message)

    def info(self, *messages):
        message = ''
        for m in messages:
            message += str(m)
        self.logger.info(message)



"""
logger = Logger(writetofile = True)

logger.info('this is a test log')

ten = torch.tensor([0.00,1.00],dtype=float,requires_grad=True)

logger.info('test',ten,9,9,9)
"""