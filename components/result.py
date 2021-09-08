import pickle

class SynthResult:
    def __init__(self, programs, num_exec, time, eval_result):
        self.programs = programs
        self.num_exec = num_exec
        self.time = time
        self.eval_result = eval_result

class DecodeResult:
    def __init__(self, programs, eval_result, num_exec=0):
        self.decodes = programs
        self.eval_result = eval_result
        self.num_exec = num_exec
