"""Waitress configuration for Windows production deployment."""

from waitress import serve
from app import app, initialize_sam

def main():
    """Run the application with Waitress server."""
    print("Initializing SAM model...")
    initialize_sam()
    
    print("Starting Waitress server on http://0.0.0.0:9487")
    print("Access the web interface at: http://localhost:9487")
    
    serve(
        app,
        host='0.0.0.0',
        port=9487,
        threads=4,
        connection_limit=100,
        cleanup_interval=30,
        channel_timeout=300,
        max_request_body_size=104857600,  # 100MB
        expose_tracebacks=False
    )

if __name__ == '__main__':
    main() 