import sys
import os
import datetime


def generate_timestamp(fmt="%Y%m%d_%H%M%S"):
    return datetime.datetime.now().strftime(fmt)


def print_log(msg_string, msg_tag=''):
    print("{}: {} {}".format(generate_timestamp(fmt="%Y-%m-%d %H:%M:%S.%f"), msg_tag, msg_string))


def override_stdout_stderr(logs_dir, log_tag):
    timestamp = generate_timestamp()
    logfile_path = os.path.join(logs_dir, "{}_{}.log".format(timestamp, log_tag))
    logfile = open(logfile_path, 'w+')
    print("logfile path: {}".format(logfile_path))
    return stream_tee(sys.stdout, logfile), stream_tee(sys.stderr, logfile), timestamp


class stream_tee(object):
    # Based on https://gist.github.com/327585 by Anand Kunal
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None  # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name  # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        callable2 = getattr(self.stream2, self.__missing_method_name)
        callable2(*args, **kwargs)

        # Emit method call to stdout (stream 1)
        callable1 = getattr(self.stream1, self.__missing_method_name)
        return callable1(*args, **kwargs)


if __name__ == '__main__':
    logfile = open("blah.txt", "w+")

    sys.stdout = stream_tee(sys.stdout, logfile)

    print(generate_timestamp())
    print("# Now, every operation on sys.stdout is also mirrored on logfile")

    logfile.close()
