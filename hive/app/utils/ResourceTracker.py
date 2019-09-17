import psutil


class ResourceTracker:
    """
    Implements a switcher that receives strings and dynamically chooses the psutil functions that should be called
    returning the appropriate results.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_cpu_usage():
        return psutil.cpu_percent(interval=0, percpu=False)

    @staticmethod
    def get_cpu_count():
        return psutil.cpu_count(logical=False)

    @staticmethod
    def get_cpu_avg():
        return psutil.getloadavg()

    @staticmethod
    def get_mem_usage():
        return psutil.virtual_memory()

    @staticmethod
    def get_disk_info():
        data = dict.fromkeys(['disk_total', 'disk_used', 'disk_percent'], 0.0)
        mounted_disk_partitions = psutil.disk_partitions()
        for partition in mounted_disk_partitions:
            if partition:
                partition_stats = psutil.disk_usage(partition)
                data['total'] += ResourceTracker.bytes_to_gigabytes(partition_stats.total)
                data['used'] += ResourceTracker.bytes_to_gigabytes(partition_stats.used)
        try:
            data['percent'] = data['used'] / data['total']
        except ZeroDivisionError:
            pass
        return data

    @staticmethod
    def get_hardware_temp():
        return psutil.sensors_temperatures(fahrenheit=False)

    SWITCHER = {
        'cpu': get_cpu_usage,
        'cpu_count': get_cpu_count,
        'cpu_avg': get_cpu_avg,
        'mem': get_mem_usage,
        'disk': get_disk_info,
        'temp': get_hardware_temp
    }

    @staticmethod
    def get_value(resource):
        """
        :param resource: the name of the resource you wish to get utilization data from
        :type str
        :return: requested stats as float, int, list, dictionary or other object types depending on requested stat
        """
        try:
            return ResourceTracker.SWITCHER.get(resource, -1)()
        except psutil.Error:
            return -1

    @staticmethod
    def bytes_to_kilobytes(val):
        return float("{0:.2f}".format(val/1024.0))

    @staticmethod
    def bytes_to_megabytes(val):
        return float("{0:.2f}".format(val/1024/1024.0))

    @staticmethod
    def bytes_to_gigabytes(val):
        return float("{0:.2f}".format(val/1024/1024/1024.0))
