# -*- coding: utf-8 -*-
import sys
import binascii
import codecs

if __name__ == "__main__":
    msg = "헬로월드"

    chrset = sys.getdefaultencoding()
    print(msg)
    print(len(msg))
    print(msg.encode())
    #print(msg.encode('hex'))
    print(b'\xed\x97\xac\xeb\xa1\x9c\xec\x9b\x94\xeb\x93\x9c'.decode())
    print(binascii.hexlify(b'\xed\x97\xac\xeb\xa1\x9c\xec\x9b\x94\xeb\x93\x9c'))
    print(binascii.hexlify(msg.encode()).decode())
    print(msg.encode().hex())
    print(codecs.encode(msg.encode(), 'hex'))
