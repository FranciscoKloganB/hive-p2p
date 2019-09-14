from domain import CryptoUtils
from domain import ConvertUtils


class SharedFilePart:
    def __init__(self, part_id, part_data):
        self.part_id = part_id
        self.part_data = ConvertUtils.bytes_to_base64_string(part_data)
        self.sha256 = CryptoUtils.sha256(part_data)
