import functools
class OutputRecorder:
    def __init__(self, record_all=True):
        self.record_all = record_all
        self.recorded_outputs = {}

    def record_output(self, func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.record_all or self._should_record_function(func.__module__, func.__qualname__):
                output = func(self, *args, **kwargs)
                key = f"{func.__module__}.{func.__qualname__}"
                self.recorded_outputs[key] = output
            else:
                output = func(self, *args, **kwargs)
            return output

        return wrapper

    def get_output(self, module_name, func_name):
        key = f"{module_name}.{func_name}"
        return self.recorded_outputs.get(key)

    def _should_record_function(self, module_name, func_name):
        # Override this method to implement custom logic for selective recording
        return True

global_output_recorder = OutputRecorder()