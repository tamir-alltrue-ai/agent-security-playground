from mcp.server.fastmcp import FastMCP

app = FastMCP()

@app.tool()
def add(a: int, b: int) -> int:
    """
    Adds two integers together.
    """
    return a + b


if __name__ == '__main__':
    app.run(transport='stdio')
