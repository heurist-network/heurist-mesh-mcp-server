# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Set the working directory
WORKDIR /app

# Copy dependency files first (for better caching)
COPY uv.lock pyproject.toml README.md ./

# Install dependencies with caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

# Copy the rest of the application files
COPY . .

# Set the link mode to copy and compile bytecode
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Default environment variables
ENV PORT=8000
ENV TRANSPORT=sse

# Use ENTRYPOINT with sh -c to allow variable expansion
ENTRYPOINT ["sh", "-c", "uv run mesh-tool-server --transport ${TRANSPORT} --port ${PORT}"]