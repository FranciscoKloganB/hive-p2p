from domain import CryptoUtils


class SharedFilePart:
    def __init__(self, part_id, part_data):
        self.part_id = part_id
        self.part_data = part_data
        self.sha256 = CryptoUtils.sha256(part_data)
