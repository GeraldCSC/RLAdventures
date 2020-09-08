accepted_kwargs = {"state", "action", "log_prob", "value", "reward", "done"}

class PGMemory():
    def __init__(self):
        self.buf = {}

    def clear(self):
        self.__init__()

    def check_key(self, k):
        if k not in accepted_kwargs:
            raise Exception("invalid key to push object into")

    def push(self, **kwargs):
        for k,v in kwargs.items():
            self.check_key(k)
            if k not in self.buf:
                self.buf[k] = []
            self.buf[k].append(v)

    def get(self, *args):
        ret_list = []
        for k in args:
            self.check_key(k)
            ret_list.append(self.buf[k])
        return ret_list
