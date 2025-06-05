import base64
import email
import logging
import os
import sys
from email.message import EmailMessage
from typing import List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
ME = "John Dow"
TEST_EMAIL = os.environ.get("TEST_EMAIL", "N/A")


class Email:
    def __init__(self, thread_id: str, email_id: str, unread: bool, message_raw):
        self.thread_id = thread_id
        self.email_id = email_id
        self.unread = unread
        email_message = email.message_from_bytes(base64.urlsafe_b64decode(message_raw["raw"]))
        self.mail_from = email_message.get("from")
        self.subject = email_message.get("subject")
        self.message_id = email_message.get("Message-ID")
        self.body = f"{self.subject}\n"

        if email_message.is_multipart():
            payload = email_message.get_payload()
            text_body = None
            html_body = None
            for p in payload:
                content_type = p.get_content_type()
                if content_type == "text/plain":
                    text_body = p.get_payload(decode=True)
                elif content_type == "text/html":
                    html_body = p.get_payload(decode=True)
            self.body += (text_body if text_body else html_body).decode("utf-8")
        else:
            self.body += email_message.get_payload(decode=True).decode("utf-8")


class EmailThread:
    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.last_customer_email = None
        self.email_list = []
        self.unread_email_ids = []

    def add_email(self, email_message: Email, is_my_email: bool):
        self.email_list.append(email_message)
        if email_message.unread:
            self.unread_email_ids.append(email_message.email_id)
            if not is_my_email:
                self.last_customer_email = email_message


class GmailHelper:
    def __init__(self, label: str = None):
        self.service = self.get_gmail_service()
        self.account = self.get_account_email()
        self.labels = [self.get_label_id(label)]

    @staticmethod
    def get_gmail_service():
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return build("gmail", "v1", credentials=creds)

    def get_account_email(self) -> str:
        profile = self.service.users().getProfile(userId="me").execute()
        return profile["emailAddress"]

    def is_my_reply_email(self, email_message: Email) -> bool:
        return self.account in email_message.mail_from

    def get_label_id(self, label_name) -> str:
        if not label_name or label_name == "INBOX":
            return "INBOX"

        results = self.service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        for label in labels:
            if label_name == label["name"]:
                return label["id"]

        return "INBOX"

    def mark_as_read(self, message_id):
        body = {"removeLabelIds": ["UNREAD"]}
        self.service.users().messages().modify(userId="me", id=message_id, body=body).execute()

    def reply_email(self, email_message: Email, content, draft=False):
        if self.is_my_reply_email(email_message):
            logging.warning("Send to self should not be allowed.")
            return

        message = EmailMessage()
        message.set_content(content)
        message["To"] = email_message.mail_from
        message["From"] = f"{ME} <{self.account}>"
        message["Subject"] = email_message.subject
        message["References"] = email_message.message_id
        message["In-Reply-To"] = email_message.message_id

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        if draft:
            create_message = {"message": {"raw": encoded_message, "threadId": email_message.thread_id}}
            self.service.users().drafts().create(userId="me", body=create_message).execute()
        else:
            create_message = {"raw": encoded_message, "threadId": email_message.thread_id}
            self.service.users().messages().send(userId="me", body=create_message).execute()

        logging.info("Replay email sent")

    def get_email_threads(self) -> List[EmailThread]:
        result = (
            self.service.users()
            .messages()
            .list(userId="me", q="is:unread category:primary", labelIds=self.labels)
            .execute()
        )
        messages = result.get("messages", [])
        if not messages:
            return []

        thread_ids = set()
        unread = set()
        for message in messages:
            unread.add(message["id"])
            thread_ids.add(message["threadId"])

        email_threads = []
        for thread_id in thread_ids:
            email_thread = EmailThread(thread_id)
            email_threads.append(email_thread)

            threads = self.service.users().threads().get(userId="me", id=thread_id).execute()
            for message in threads.get("messages", []):
                if "TRASH" in message.get("labelIds", []):
                    continue

                email_id = message["id"]
                message_raw = self.service.users().messages().get(userId="me", id=email_id, format="raw").execute()
                email_message = Email(thread_id, email_id, email_id in unread, message_raw)
                is_my_email = self.is_my_reply_email(email_message)
                email_thread.add_email(email_message, is_my_email)

        return email_threads


def main():
    try:
        service = GmailHelper()
        threads = service.get_email_threads()

        for email_thread in threads:
            content = ""
            for email_message in email_thread.email_list:
                if service.is_my_reply_email(email_message):
                    content += f"\nMe: {email_message.body}"
                else:
                    content += f"\nCustomer: {email_message.body}"

            logging.info(f"Aggregated email thread content: {content}.")

            email_message = email_thread.last_customer_email
            if email_message and TEST_EMAIL in email_message.mail_from:
                # send response to last unread customer email
                service.reply_email(email_message, "echo")

                # mark all emails in the thread as read
                for email_id in email_thread.unread_email_ids:
                    service.mark_as_read(email_id)

        logging.info("Finished processing unread emails.")
    except HttpError as error:
        logging.error(f"An error occurred: {error}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    main()
