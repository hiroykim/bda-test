import hashlib
import sys


def md5_test(msg):
    en_msg = msg.encode()
    data = hashlib.md5(en_msg).hexdigest()

    print("========md5 test=========")
    print("md5_msg : " + msg)
    print("md5_md  : " + data)
    print("md5_md len  : " , len(data) , len(data)*8/2)

def sha1_test(msg):
    en_msg = msg.encode()
    data = hashlib.sha1(en_msg).hexdigest()

    print("========sha1 test=========")
    print("sha1_msg : " + msg)
    print("sha1_md  : " + data)
    print("sha1_md len  : " , len(data) , len(data)*8/2)



def sha256_test(msg):
    en_msg = msg.encode()
    data = hashlib.sha256(en_msg).hexdigest()
    b_data = hashlib.sha256(en_msg).digest()

    print("========sha256 test=========")
    print("sha256_msg      : " + msg)
    print("sha256_hexa_md  : " + data)
    print("sha256_hexa str, bit : " , len(data) , len(data)*8/2)
    print("sha256_b_data  : " ,  b_data.__str__() )


def sha512_test(msg):
    en_msg = msg.encode()
    h = hashlib.sha512()
    h.update(en_msg)
    h.update("def".encode())
    h.digest()
    print("========sha512 test=========")
    print("sha512_msg : " + msg)
    print("sha512_md  : " + h.hexdigest())
    print("sha512_md len  : " , h.digest_size , h.block_size)

if __name__ == "__main__":
    msg = "abc"
    md5_test(msg)
    sha1_test(msg)
    sha256_test(msg)
    sha512_test(msg)
