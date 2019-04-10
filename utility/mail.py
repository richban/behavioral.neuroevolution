import smtplib
import os
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def report(text, img=None):
    user = os.environ['EMAIL_USER']
    password = os.environ['EMAIL_PASSWORD']

    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = user
    msg['Subject'] = 'Behavioral Neuroevolution Reporting'

    text = MIMEText(''.join(text))
    msg.attach(text)

    if img:
        img_data = open(img, 'rb').read()
        image = MIMEImage(img_data, name=os.path.basename(img))
        msg.attach(image)

    try:
        with smtplib.SMTP('smtp.office365.com', 587, timeout=10) as server_ssl:
            server_ssl.starttls()
            server_ssl.login(user, password)
            server_ssl.sendmail(msg['From'], msg['To'], msg.as_string())
    except Exception as ex:
        print(ex.__class__.__name__ + ": " + 'Email Not Sent!')
