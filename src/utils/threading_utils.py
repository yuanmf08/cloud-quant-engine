from threading import Thread


class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.ret


def processing_threads(target, args: list):
    """
    :param target: function
    :param args: list of set of args
    """
    thread_list = []
    for item in args:
        thread_list.append(
            PropagatingThread(
                target=target,
                args=item
            )
        )
    for thread in thread_list:
        thread.daemon = True
        thread.start()

    returns = []
    for thread in thread_list:
        returns.append(thread.join())

    return returns
