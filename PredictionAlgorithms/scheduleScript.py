import schedule
import time

def job(t):
    print ("I'm working...", t)

schedule.every().day.at("01:00").do(job,'It is 01:00')

while True:
    schedule.run_pending()
    time.sleep(60)