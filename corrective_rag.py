import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Enterprise Knowledge Base", layout="wide")
st.title("Enterprise Knowledge Base")


def api_request(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("access_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.request(
        method,
        f"{API_URL}{path}",
        headers=headers,
        timeout=60,
        **kwargs,
    )
    if response.status_code == 401:
        st.session_state.pop("access_token", None)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(str(detail))
    return response.json() if response.content else None


if "access_token" not in st.session_state:
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        try:
            token = api_request(
                "POST",
                "/api/v1/auth/login",
                json={"username": username, "password": password},
            )
            st.session_state.access_token = token["access_token"]
            st.rerun()
        except Exception as exc:
            st.error(f"Login failed: {exc}")
    st.stop()

try:
    current_user = api_request("GET", "/api/v1/auth/me")
except Exception as exc:
    st.error(f"Unable to load the current user: {exc}")
    st.stop()

with st.sidebar:
    st.subheader(current_user["username"])
    st.caption(current_user["role"])
    st.caption(", ".join(item["name"] for item in current_user["departments"]))
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

documents_tab, query_tab, history_tab = st.tabs(
    ["Documents", "Ask", "Conversations"]
)

with documents_tab:
    if current_user["role"] in {"admin", "editor"}:
        with st.form("upload_document"):
            file = st.file_uploader("Local document", type=["pdf", "txt", "md"])
            title = st.text_input("Title")
            visibility = st.selectbox(
                "Visibility", ["department", "restricted"]
            )
            department_options = {
                item["name"]: item["uuid"]
                for item in current_user["departments"]
            }
            department_name = st.selectbox(
                "Department", list(department_options) or ["No department"]
            )
            submitted = st.form_submit_button("Upload", type="primary")
        if submitted:
            if file is None:
                st.warning("Choose a local PDF, TXT, or Markdown file.")
            elif not department_options:
                st.error("Your account has no department.")
            else:
                try:
                    result = api_request(
                        "POST",
                        "/api/v1/documents",
                        files={
                            "file": (
                                file.name,
                                file.getvalue(),
                                file.type or "application/octet-stream",
                            )
                        },
                        data={
                            "title": title,
                            "visibility": visibility,
                            "department_uuid": department_options[department_name],
                            "acl_user_uuids": "[]",
                            "acl_department_uuids": "[]",
                        },
                    )
                    st.session_state.last_job_uuid = result["job_uuid"]
                    st.success("Upload accepted. Processing has started.")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    job_uuid = st.session_state.get("last_job_uuid")
    if job_uuid:
        try:
            job = api_request("GET", f"/api/v1/jobs/{job_uuid}")
            st.progress(job["progress"], text=f"{job['stage']} ({job['status']})")
            if job["status"] not in {"ready", "failed"}:
                if st.button("Refresh processing status"):
                    st.rerun()
            elif job["status"] == "failed":
                st.error(job.get("error_message") or "Processing failed")
        except Exception as exc:
            st.warning(f"Unable to read job status: {exc}")

    try:
        documents = api_request("GET", "/api/v1/documents")
        for document in documents:
            version = document.get("current_version")
            with st.container(border=True):
                st.subheader(document["title"])
                st.caption(
                    f"{document['department']['name']} | "
                    f"{document['visibility']} | "
                    f"{'ready' if version else 'processing'}"
                )
                if version:
                    st.write(
                        f"Version {version['version_number']}, "
                        f"{version.get('chunk_count') or 0} chunks"
                    )
    except Exception as exc:
        st.error(f"Unable to list documents: {exc}")

with query_tab:
    with st.form("question"):
        question = st.text_area("Question", height=120)
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        try:
            with st.spinner("Searching authorized knowledge..."):
                result = api_request(
                    "POST",
                    "/api/v1/queries",
                    json={"question": question},
                )
            if result["refused"]:
                st.warning(result["answer"])
            else:
                st.markdown(result["answer"])
            for index, citation in enumerate(result["citations"], start=1):
                with st.expander(
                    f"[{index}] {citation['document_title']} "
                    f"v{citation['version']}"
                ):
                    st.caption(
                        f"Page {citation.get('page_number') or '-'} | "
                        f"{citation['chunk_id']}"
                    )
                    st.write(citation["excerpt"])
            st.caption(
                f"Trace: {result.get('trace_id')} | "
                f"Total: {result['timings'].get('total', 0):.2f}s"
            )
        except Exception as exc:
            st.error(f"Question failed: {exc}")

with history_tab:
    try:
        conversations = api_request("GET", "/api/v1/conversations")
        for conversation in conversations:
            with st.expander(conversation["title"]):
                for message in conversation["messages"]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
    except Exception as exc:
        st.error(f"Unable to load conversations: {exc}")
