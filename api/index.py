"""Vercel entry point.

This file exposes the FastAPI `app` object required by the @vercel/python runtime.
It simply imports the existing application defined in `services/geoclustering.py`.
"""

from services.geoclustering import app as fastapi_app


app = fastapi_app
