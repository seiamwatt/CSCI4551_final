import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eiamw001/CSCI4511_project/install/security_coverage_robot'
