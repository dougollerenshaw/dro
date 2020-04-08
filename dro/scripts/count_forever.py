import time

count = 71740

while True:
    print(' If Elliott was counting, he would be at the number {}         '.format(count), end='\r')
    count = count + 1
    time.sleep(1)