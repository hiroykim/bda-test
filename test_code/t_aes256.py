# 실행 환경
# windows10, python3.6.8, pycharm(2020.03.05)
import base64
import hashlib
#pip install pycryptodomex 또는 pycryptodome
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes

# 암호화할 문자열을 일정 크기로 나누기 위해서, 모자란 경우 크기를 채워줍니다.
BS = 16
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS).encode()
unpad = lambda s: s[0:-s[-1]]


# 암호화를 담당할 클래스 입니다.
class AESCipher:

    # 클래스 초기화 - 전달 받은 키를 해시 값으로 변환해 키로 사용합니다.
    def __init__(self):
        self.key = hashlib.sha256("01234567890123456789012345678901".encode()).digest()

    def encrypt_good(self, text):
        text = text.encode()
        text = pad(text)
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(text)).decode()

    def encrypt(self, text):
        text = text.encode()
        text = pad(text)
        iv = self.key[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(text)).decode()

    # 복호화 - 전달 받은 값을 복호화 한후, 언패딩해 원문을 전달합니다.
    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(enc[AES.block_size:])).decode()

def good_iv(text):
    encrypted = aes256.encrypt(text)
    print("==========GOOD Start==========")
    print("입력   :" + text)
    print("암호화 :" + aes256.encrypt_good(text))
    print("복호화 :" + aes256.decrypt(encrypted))

def bad_iv(text):
    encrypted = aes256.encrypt(text)
    print("==========BAD Start==========")
    print("입력   :" + text)
    print("암호화 :" + aes256.encrypt(text))
    print("복호화 :" + aes256.decrypt(encrypted))

if __name__ == "__main__":

    aes256 = AESCipher()
    text = "!! 안녕하세요 고객님 !!";

    bad_iv(text)
    bad_iv(text)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    good_iv(text)
    good_iv(text)
