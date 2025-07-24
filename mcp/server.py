"""
FastMCP quickstart example.

cd to the `examples/snippets/clients` directory and run:
    uv run server fastmcp_quickstart stdio
"""
import re
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from git import Repo
from diffops import FuncLevelDiffGenerator


# Create an MCP server
mcp = FastMCP("output-formatters")


class RepoCloneResponse(BaseModel):
    """Response from the RepoCloner tool."""
    repo_path: str = Field(description="The absolute path to the cloned repo.")
    clone_success: bool = Field(description="Whether the repo was cloned successfully.")

class CommitExistsResponse(BaseModel):
    """Response from the CommitExists tool."""
    commit_exists: bool = Field(description="Whether the commit exists in the repository.")
    error_msg: str = Field(description="The error message if the commit does not exist.")


class Vulnerability(BaseModel):
    """Vulnerability structure."""
    file_path: str = Field(description="The path to the file containing the vulnerable function, relative to the repository root. Do not include any absolute or user-specific paths.")
    function_name: str = Field(description="The fully-qualified name of the vulnerable function.")
    commit_sha: str = Field(description="The SHA of the commit that fixed the vulnerability.")
    diff_url: str = Field(description="The URL of the diff that fixed the vulnerability.")
    affected_versions: List[str] = Field(description="The versions of the library that are affected by the vulnerability.")


@mcp.tool(name="VulnerableFunctionSearchFormatter")
def vulnerable_function_presentation(vulnerabilities: List[Vulnerability]) -> str:
    """Formats CVE analysis results into a clear, structured report that highlights each vulnerable function and pinpoints its exact location in the codebase.
    
    Raises:
        ValueError: If the input formatting has any issues.
    """
    if len(vulnerabilities) < 1:
        return "Input formatting is incorrect. Please provide a list of vulnerabilities."

    for vuln in vulnerabilities:
        if vuln.file_path == "":
            raise ValueError("File path is empty. Please provide a valid file path for the vulnerable function.")
        else:
            if vuln.file_path.startswith(str(Path.home())):
                raise ValueError(f"Possible security issue: The file path ({vuln.file_path}) contains a user-specific path. Do not include any absolute or user-specific paths.")

        if vuln.function_name == "":
            raise ValueError("Function name is empty. Please provide a valid function name for the vulnerable function.")

        if vuln.commit_sha == "":
            raise ValueError("Commit SHA is empty. Please provide a valid commit SHA for the vulnerable function.")
        else:
            # Check if the commit sha is valid (is hexadecimal and between 7-40 characters long)
            if not re.match(r'^[a-fA-F0-9]{7,40}$', vuln.commit_sha):
                raise ValueError(f"Commit SHA ({vuln.commit_sha}) is invalid. Please provide the valid commit SHA for the vulnerable function. It should be a hexadecimal string between 7-40 characters long.")
        
        if vuln.diff_url == "":
            raise ValueError("Diff URL is empty. Please provide a valid diff URL for the vulnerable function.")
        
        if vuln.affected_versions == []:
            raise ValueError("Affected versions is empty. Please provide a valid list of affected versions for the vulnerable function.")
    
    return "Input formatting is correct. Please proceed with the analysis."


@mcp.tool(name="RepoCloner")
def repo_cloner(repo_slug: str, host: str = "github") -> RepoCloneResponse:
    """Clones a repository and returns the absolute path to the cloned repo.
    Always use this tool to clone repos that are on Github or Gitlab. Repos hosted on other platforms are not supported.
    If this tool fails, fallback to clone the repo manually using the repo_url and git clone.
    
    Args:
        repo_slug: The slug of the repository to clone.
        host: The web-based platform where the Git repo is hosted.
    Returns:
        The absolute path to the cloned repo.
    """
    cache = Path("~/sandbox/temp").expanduser()
    if not cache.exists():
        cache.mkdir(parents=True, exist_ok=True)

    try:        
        with FuncLevelDiffGenerator.create(repo_slug, repo_cache=cache, host=host) as generator:
            return RepoCloneResponse(repo_path=generator.repo_path, clone_success=True)
    except Exception as e:
        return RepoCloneResponse(repo_path="", clone_success=False)


@mcp.tool(name="CommitExists")
def commit_exists(repo_path: str, commit_sha: str) -> CommitExistsResponse:
    """Checks if a commit exists in a repository."""
    try:
        Repo(repo_path).commit(commit_sha)
        return CommitExistsResponse(commit_exists=True, error_msg="")
    except Exception as e:
        return CommitExistsResponse(commit_exists=False, error_msg=str(e))


def main():
    """Entry point for the direct execution server."""
    print("Starting MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()