# Enterprise Incident Intelligence

This app helps a company track incidents and decide what to do next.

## What this app does

- Lets a worker enter a problem report
- Saves the report in a local database
- Gives a category for the problem
- Suggests how urgent the problem is
- Writes a short summary of the problem

## Who uses it

- A worker in the company enters the incident.
- This can be support staff, IT staff, or a manager.
- If a customer tells support about the issue, support enters it in the app.
- A manager or worker then looks at the incident and decides what to do.

## Example incidents

- “The website is down and users cannot log in.”
- “The payment page shows error 502.”
- “Email notifications are not being sent.”
- “The app is very slow today.”
- “We saw a security alert on the admin panel.”

## How a manager uses it

- Open the app in a browser
- See the list of incidents
- Read the AI category, priority, and summary
- Pick the most urgent incident to fix first

## How it works

1. The browser form sends the incident to the backend.
2. The backend saves it in `incidents.db`.
3. The AI engine creates a category, priority, and summary.
4. The app shows the incident and AI results.

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
.venv\Scripts\activate
```

3. Install the packages:

```bash
python -m pip install -r requirements.txt
```

4. Run the app:

```bash
python main.py
```

5. Open the browser at `http://127.0.0.1:8000`

## Packages and why they are used

- `fastapi` — makes the backend web app.
- `uvicorn` — runs the web server.
- `sentence-transformers` — helps the AI understand incident text.
- `transformers` — supports AI text tools.
- `torch` — runs the AI code.
- `

## Notes

- The app uses free open-source tools.
- Incident data stays saved after restart.
- The AI runs on the computer CPU.
