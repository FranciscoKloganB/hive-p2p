import unittest
import json
from domain.SharedFilePart import SharedFilePart


class SharedFilePartCreationAndJSONSerialization(unittest.TestCase):
    def test_json_creation(self):
        sfp1 = SharedFilePart("partIdTest01", "hello")
        sfp1_d = sfp1.__dict__
        print(sfp1_d)

        with open("jsontest01.json", "w") as j1:
            json.dump(sfp1_d, j1)

        sfp2 = SharedFilePart("partIdTest02", b"hello")
        sfp2_d = sfp2.__dict__
        print(sfp2_d)

        with open("jsontest02.json", "w") as j2:
            json.dump(sfp2_d, j2)

            
if __name__ == '__main__':
    unittest.main()
