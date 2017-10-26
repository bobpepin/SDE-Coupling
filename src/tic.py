import time
class Tic(object):
    def __enter__(self):
        self.tic = time.perf_counter()

    def __exit__(self, _1, _2, _3):
        self.toc = time.perf_counter()
        self.delta = self.toc - self.tic
        print("Elapsed time is %.2f seconds." % (self.delta))
tic = Tic()
