import multiprocessing
import traceback

def run_plugin_in_sandbox(plugin_cls, *args, **kwargs):
    def target(queue, *args, **kwargs):
        try:
            plugin = plugin_cls()
            result = plugin.run(*args, **kwargs)
            queue.put(("success", result))
        except Exception as e:
            queue.put(("error", traceback.format_exc()))
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(queue,)+args, kwargs=kwargs)
    p.start()
    p.join(timeout=30)  # Timeout for safety
    if not queue.empty():
        status, data = queue.get()
        if status == "success":
            return data
        else:
            raise RuntimeError(f"Plugin error: {data}")
    else:
        raise TimeoutError("Plugin execution timed out.") 