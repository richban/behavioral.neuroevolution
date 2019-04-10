import smtplib
import os

__credits__ = 'vladas-v'

def report_gmail(text):
    gmail_user = os.environ['GMAIL_USER']
    app_pass = os.environ['GMAIL_PASSWORD']

    sent_from = gmail_user
    to = gmail_user
    subject = 'Behavioral Neuroevolution Reporting'
    body = ''.join(text)

    email_text = "" \
                 "From: {sent_from}\n" \
                 "To: {to}\n" \
                 "Subject: {subject}\n" \
                 "\n{body}".format(sent_from=sent_from,
                                   to=to,
                                   subject=subject,
                                   body=body)
    try:
        server_ssl = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server_ssl.ehlo()   # optional
        server_ssl.login(gmail_user, app_pass)
        server_ssl.sendmail(sent_from, to, email_text)
        server_ssl.close()
    except Exception:
        pass
