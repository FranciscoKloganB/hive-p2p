import hashlib


class SharedFilePart:
    def __init__(self, part_id, part_data):
        self.part_id = part_id
        self.part_data = part_data
        self.sha256 = __get_sha_256(part_data)

    @staticmethod
    def __get_sha_256(part_data):
        return hashlib.sha256(part_data).hexdigest()
