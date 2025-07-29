"""
Contract implementation for extracting per-function unified diffs.

This module provides functionality to generate function-level unified diffs 
(including the full, outermost function context) from a Git commit.

The main contract:
- Inputs: repo_slug, parser (Tree-sitter), commit_hash
- Process: Clone repo, find changed files, compare commit~1 vs commit, extract functions
- Output: JSON array of function diffs for all changed functions across all files
"""

import re
import os
import tempfile
import shutil
import fnmatch
import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass
import difflib
from pathlib import Path
from unidiff import PatchSet
import git
from git import Repo
from tqdm import tqdm


@dataclass
class LanguageConfig:
    """Configuration for language-specific AST parsing."""
    name: str
    function_node_types: Set[str]
    class_node_types: Set[str]
    identifier_node_type: str = "identifier"
    qualified_name_separator: str = "."
    
    @classmethod
    def python(cls) -> 'LanguageConfig':
        """Configuration for Python."""
        return cls(
            name="python",
            function_node_types={"function_definition", "async_function_definition"},
            class_node_types={"class_definition"},
            identifier_node_type="identifier",
            qualified_name_separator="."
        )
    
    @classmethod
    def javascript(cls) -> 'LanguageConfig':
        """Configuration for JavaScript/TypeScript."""
        return cls(
            name="javascript",
            function_node_types={"function_declaration", "function_expression", "arrow_function", "method_definition"},
            class_node_types={"class_declaration"},
            identifier_node_type="identifier",
            qualified_name_separator="."
        )
    
    @classmethod
    def java(cls) -> 'LanguageConfig':
        """Configuration for Java."""
        return cls(
            name="java",
            function_node_types={"method_declaration", "constructor_declaration"},
            class_node_types={"class_declaration", "interface_declaration"},
            identifier_node_type="identifier",
            qualified_name_separator="."
        )
    
    @classmethod
    def c_cpp(cls) -> 'LanguageConfig':
        """Configuration for C/C++."""
        return cls(
            name="c_and_cpp",
            function_node_types={"function_definition", "function_declarator"},
            class_node_types={"class_specifier", "struct_specifier"},
            identifier_node_type="identifier",
            qualified_name_separator="::"
        )
    
    @classmethod
    def csharp(cls) -> 'LanguageConfig':
        """Configuration for C#."""
        return cls(
            name="csharp",
            function_node_types={"method_declaration", "constructor_declaration"},
            class_node_types={"class_declaration", "interface_declaration", "struct_declaration"},
            identifier_node_type="identifier",
            qualified_name_separator="."
        )
    
    @classmethod
    def rust(cls) -> 'LanguageConfig':
        """Configuration for Rust."""
        return cls(
            name="rust",
            function_node_types={"function_item"},
            class_node_types={"struct_item", "enum_item", "impl_item"},
            identifier_node_type="identifier",
            qualified_name_separator="::"
        )
    
    @classmethod
    def go(cls) -> 'LanguageConfig':
        """Configuration for Go."""
        return cls(
            name="go",
            function_node_types={"function_declaration", "method_declaration"},
            class_node_types={"type_declaration"},  # Go doesn't have classes, but has types
            identifier_node_type="identifier",
            qualified_name_separator="."
        )


@dataclass
class FunctionSpan:
    """Represents a function's location and metadata in source code."""
    name: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    class_name: Optional[str] = None
    qualified_name_separator: str = "."
    
    @property
    def qualified_name(self) -> str:
        """Return the qualified function name (ClassName.methodName or functionName)."""
        if self.class_name:
            return f"{self.class_name}{self.qualified_name_separator}{self.name}"
        return self.name


@dataclass
class DiffRanges:
    """Represents line ranges changed in a diff, separated by source and target files."""
    source_ranges: List[Tuple[int, int]]
    target_ranges: List[Tuple[int, int]]


class FuncLevelDiffGenerator:
    """
    Multi-language generator for per-function unified diffs with full context from Git repositories.
    
    This class implements the contract for extracting function-level unified diffs by:
    1. Cloning a Git repository to a temporary directory using GitPython
    2. Determining files changed in a given commit  
    3. Automatically detecting all programming languages present in the changed files
    4. Dynamically loading appropriate Tree-sitter parsers for each detected language
    5. Filtering out commits that only change non-code files (using language-specific ignore patterns)
    6. Comparing file contents between commit~1 and commit for each changed file
    7. Using Tree-sitter AST parsing to identify functions/methods in each supported language
    8. Generating unified diffs for each function that has changes across all files
    
    Automatically supports multiple programming languages with pre-configured settings for:
    - Python (.py, .pyi, .pyx) - functions, async functions, class methods
    - JavaScript/TypeScript (.js, .jsx, .ts, .tsx) - functions, arrow functions, methods
    - Java (.java) - methods, constructors  
    - C/C++ (.c, .cpp, .h, .hpp) - functions, class methods with :: separator
    - C# (.cs) - methods, constructors
    - Rust (.rs) - functions, impl methods with :: separator
    - Go (.go) - functions, methods
    
    Key Features:
    - **Multi-Language Commits**: Handles commits with files in different languages simultaneously
    - **Dynamic Parser Loading**: Only loads parsers for languages actually present in each commit
    - **Language-Specific Filtering**: Applies appropriate ignore patterns per file type
    - **Automatic Language Detection**: No need to specify languages upfront
    - **Polyglot Repository Support**: Perfect for modern repositories mixing frontend/backend/scripts
    
    Usage Examples:
    
    Basic Multi-Language Usage:
        >>> # No parsers needed upfront - they're loaded automatically!
        >>> with FuncLevelDiffGenerator.create("owner/polyglot-repo") as generator:
        ...     result = generator("abc123456")  # Call with commit hash
        ...     print(result)  # Function diffs from ALL languages in the commit
        
        >>> # With repository caching for better performance
        >>> with FuncLevelDiffGenerator.create("owner/repo", repo_cache="./cache") as generator:
        ...     result = generator("abc123456")  # Uses cached repo if available
        
        >>> # Support for GitLab repositories
        >>> with FuncLevelDiffGenerator.create("owner/repo", host="gitlab") as generator:
        ...     result = generator("abc123456")  # Clones from GitLab
        
        >>> # Example result for a commit touching Python, JavaScript, and Go:
        >>> [
        ...     {"file_path": "api/main.py", "func_name": "UserService.authenticate", 
        ...      "contextualized_changes": "...", "file_language": "python"},
        ...     {"file_path": "web/app.js", "func_name": "handleLogin", 
        ...      "contextualized_changes": "...", "file_language": "javascript"},
        ...     {"file_path": "cmd/server.go", "func_name": "main", 
        ...      "contextualized_changes": "...", "file_language": "go"}
        ... ]
    
    Context Manager (Recommended):
        >>> with FuncLevelDiffGenerator.create("owner/repo") as generator:
        ...     # Multiple operations reuse the same cloned repository
        ...     result1 = generator("commit1")  # Might load Python + JS parsers
        ...     result2 = generator("commit2")  # Might reuse + load Java parser
        ...     result3 = generator("commit3")  # Reuses all previously loaded parsers
        ...     # Automatic cleanup when exiting context
        
    Repository Caching:
        >>> # Cache repositories for faster repeated access
        >>> cache_dir = Path("./repo_cache")
        >>> with FuncLevelDiffGenerator.create("owner/repo", repo_cache=cache_dir) as gen:
        ...     result = gen("abc123456")  # First time: clones to cache/github--owner--repo
        ...     
        >>> # Subsequent uses reuse cached repository with latest changes
        >>> with FuncLevelDiffGenerator.create("owner/repo", repo_cache=cache_dir) as gen:
        ...     result = gen("def789012")  # Pulls latest, then analyzes
        
        >>> # GitLab repositories are cached separately
        >>> with FuncLevelDiffGenerator.create("owner/repo", repo_cache=cache_dir, host="gitlab") as gen:
        ...     result = gen("abc123456")  # Clones to cache/gitlab--owner--repo
    
    Commit Analysis:
        >>> generator = FuncLevelDiffGenerator.create("owner/repo")
        >>> 
        >>> # Analyze different types of commits
        >>> result = generator("abc123456")  # Specific commit hash
        >>> result = generator("HEAD")       # Latest commit
        >>> result = generator("main")       # Branch tip
        >>> result = generator("v1.2.3")     # Tagged release
        >>> 
        >>> generator.cleanup()  # Manual cleanup
    
    Real-World Examples:
        >>> # Analyzing a full-stack web application commit
        >>> with FuncLevelDiffGenerator.create("company/webapp") as gen:
        ...     result = gen("deploy-v2.1")
        ...     
        ...     for diff in result:
        ...         lang = diff["file_language"] 
        ...         file = diff["file_path"]
        ...         func = diff["func_name"]
        ...         print(f"[{lang}] {file}::{func}")
        ...         
        ...     # Output might be:
        ...     # [python] backend/api/auth.py::AuthHandler.login
        ...     # [javascript] frontend/src/components/LoginForm.jsx::handleSubmit
        ...     # [typescript] frontend/src/types/user.ts::validateUser
        ...     # [go] services/notification/main.go::SendWelcomeEmail
    
    Output Format:
        The generator returns a list of dictionaries with format:
        [
          {
            "file_path": "path/to/file.ext",
            "func_name": "ClassName.methodName" | "functionName", 
            "contextualized_changes": "<unified diff for entire function>",
            "file_language": "python" | "javascript" | "java" | etc.
          },
          ...
        ]
    
    Language Detection:
        File extensions are automatically mapped to parsers:
        - .py, .pyi, .pyx → python parser
        - .js, .jsx, .mjs → javascript parser  
        - .ts, .tsx → typescript parser
        - .java → java parser
        - .c, .h → c parser
        - .cpp, .hpp, .cc, .cxx → cpp parser  
        - .cs → c_sharp parser
        - .rs → rust parser
        - .go → go parser
    
    Language-Specific Ignore Patterns:
        Each file type gets appropriate filtering:
        - Python: Ignores __pycache__/, *.pyc, setup.py, etc.
        - JavaScript: Ignores node_modules/, .next/, package.json, etc.  
        - Java: Ignores target/, *.class, pom.xml, etc.
        - C/C++: Ignores build/, *.o, *.exe, CMakeFiles/, etc.
        - And more...
    
    Parser Requirements:
        Requires tree-sitter-languages package:
        >>> pip install tree-sitter-languages
        
        Parsers are loaded on-demand, so only install what you need.
        If a language parser is unavailable, files in that language are skipped.
    
    Error Handling:
        - Missing parsers: Language is skipped with a warning
        - Binary files: Skipped automatically  
        - Encoding issues: Individual files skipped with warning
        - Invalid commits: Raises git.exc.GitError
        - No changes found: Raises ValueError("No function diffs found")
    
    Notes:
        - Repository is cloned once during initialization for efficiency
        - Uses GitPython for robust Git operations
        - Repository caching: When repo_cache is provided, repositories are stored persistently
          and reused across sessions with automatic updates from remote main/master
        - Temporary directories are automatically cleaned up; cached repositories are preserved
        - Function names are qualified (ClassName.methodName for methods)
        - Language-specific naming conventions are respected (:: for C++/Rust, . for others)
        - Processes all supported code files changed in the commit automatically
        - Parsers are cached per instance to avoid reloading
        - Thread-safe for read operations after initialization
    """
    
    # Host URL mappings for different Git hosting providers
    HOST_URLS = {
        "github": "https://github.com",
        "gitlab": "https://gitlab.com"
    }
    
    # Gitignore-style patterns for files/directories we typically don't consider as "interesting code"
    ignore_patterns = {
        "python": {
            # Documentation directories
            "docs/**",
            "doc/**", 
            "documentation/**",
            "**/docs/**",
            "**/doc/**",
            "**/documentation/**",
            
            # CI/CD directories
            ".github/**",
            ".gitlab/**",
            ".circleci/**",
            "ci/**",
            "**/ci/**",
            
            # Test directories (flexible patterns)
            "tests/**",
            "test/**", 
            "testing/**",
            "*tests/**",
            "*test/**",
            "*testing/**",
            "*/tests/**",
            "*/test/**",
            "**/tests/**",
            "**/test/**",
            "**/testing/**",
            
            # Python cache directories
            "__pycache__/**",
            "**/__pycache__/**",
            "*.pyc",
            "**/*.pyc",
            
            # Example/demo directories
            "examples/**",
            "example/**",
            "samples/**",
            "sample/**",
            "demos/**",
            "demo/**",
            "tutorials/**",
            "tutorial/**",
            "guides/**",
            "guide/**",
            "**/examples/**",
            "**/example/**",
            "**/demos/**",
            "**/demo/**",
            
            # Asset directories
            "assets/**",
            "static/**",
            "media/**",
            "images/**",
            "img/**",
            "pictures/**",
            "resources/**",
            "**/assets/**",
            "**/static/**",
            "**/media/**",
            "**/images/**",
            
            # Build/deployment directories
            "build/**",
            "dist/**",
            "packaging/**",
            "deploy/**",
            "deployment/**",
            "infrastructure/**",
            "infra/**",
            "docker/**",
            "**/build/**",
            "**/dist/**",
            
            # Other utility directories
            "fixtures/**",
            "data/**",
            "benchmarks/**",
            "benchmark/**",
            "performance/**",
            "perf/**",
            "sandbox/**",
            "playground/**",
            "templates/**",
            "template/**",
            "**/fixtures/**",
            "**/benchmarks/**",
            "**/templates/**"
        },
        "javascript": {
            # Node.js directories
            "node_modules/**",
            "**/node_modules/**",
            
            # Documentation directories
            "docs/**",
            "doc/**", 
            "documentation/**",
            "**/docs/**",
            "**/doc/**",
            "**/documentation/**",
            
            # CI/CD directories
            ".github/**",
            ".gitlab/**",
            ".circleci/**",
            "ci/**",
            "**/ci/**",
            
            # Test directories
            "tests/**",
            "test/**",
            "testing/**",
            "*tests/**",
            "*test/**",
            "*/tests/**",
            "*/test/**",
            "**/tests/**",
            "**/test/**",
            "**/testing/**",
            "__tests__/**",
            "**/__tests__/**",
            
            # Build/deployment directories
            "build/**",
            "dist/**",
            "out/**",
            ".next/**",
            ".nuxt/**",
            "**/build/**",
            "**/dist/**",
            
            # Example/demo directories
            "examples/**",
            "example/**",
            "demos/**",
            "demo/**",
            "**/examples/**",
            "**/demos/**",
            
            # Asset directories
            "assets/**",
            "static/**",
            "public/**",
            "**/assets/**",
            "**/static/**",
            
            # Coverage and cache
            "coverage/**",
            ".nyc_output/**",
            ".cache/**",
            "**/coverage/**"
        },
        "java": {
            # Build directories
            "target/**",
            "build/**",
            "out/**",
            "**/target/**",
            "**/build/**",
            
            # Documentation
            "docs/**",
            "doc/**",
            "javadoc/**",
            "**/docs/**",
            "**/javadoc/**",
            
            # Test directories
            "test/**",
            "tests/**",
            "testing/**",
            "**/test/**",
            "**/tests/**",
            "**/testing/**",
            
            # IDE directories
            ".idea/**",
            ".eclipse/**",
            ".vscode/**",
            
            # Example directories
            "examples/**",
            "example/**",
            "**/examples/**"
        },
         "c_and_cpp": {
             # Build System Directories
             "build/**",
             "cmake-build-*/**",
             "out/**",
             "bin/**",
             "obj/**",
             ".vs/**",
             ".vscode/**",
             ".idea/**",
             "CMakeFiles/**",
             ".deps/**",
             ".libs/**",
             "autom4te.cache/**",
             "config/**",
             "**/build/**",
             "**/bin/**",
             "**/obj/**",
             
             # Documentation
             "doc/**",
             "docs/**",
             "**/doc/**",
             "**/docs/**",
             
             # Test directories
             "test/**",
             "tests/**",
             "**/test/**",
             "**/tests/**",
             "gtest/**",
             "googletest/**",
             "catch2/**",
             
             # Dependencies
             "deps/**",
             "vcpkg/**",
             "conan/**",
             "**/deps/**"
         },
         "rust": {
             # Build directories
             "target/**",
             "**/target/**",
             
             # Documentation
             "docs/**",
             "doc/**",
             "**/docs/**",
             "**/doc/**",
             
             # Test directories  
             "tests/**",
             "**/tests/**",
             
             # Examples
             "examples/**",
             "**/examples/**"
         },
         "go": {
             # Build directories
             "bin/**",
             "pkg/**",
             "**/bin/**",
             "**/pkg/**",
             
             # Documentation
             "docs/**",
             "doc/**",
             "**/docs/**",
             "**/doc/**",
             
             # Test directories
             "*_test.go",
             "**/*_test.go",
             
             # Vendor directory
             "vendor/**",
             "**/vendor/**"
         }
    }

    # File patterns we don't consider as interesting code
    ignore_file_patterns = {
        "python": {
            # Python-specific files
            "__init__.py",
            "__about__.py", 
            "__version__.py",
            "setup.py",
            "create_version_file.py",
            "conftest.py",
            "version.py",
            
            # Documentation files
            "*.md",
            "*.mdx", 
            "*.rst",
            "*.txt",
            "README*",
            "CHANGELOG*",
            "HISTORY*",
            "NEWS*",
            
            # Configuration files
            "*.ini",
            "*.cfg",
            "*.conf",
            "*.toml",
            "*.json",
            "*.xml",
            "*.yml",
            "*.yaml",
            "*.env*",
            "*.template",
            "*.properties",
            "*.hcl",
            "*.in",
            
            # Lock and dependency files
            "*.lock",
            "Pipfile*",
            "poetry.lock",
            "requirements*.txt",
            
            # Log and data files
            "*.log",
            "*.csv",
            "*.tsv", 
            "*.dat",
            "*.sql",
            
            # Media files
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.svg",
            "*.ico",
            "*.bmp",
            "*.tiff",
            "*.webp",
            "*.pdf",
            "*.doc",
            "*.docx",
            
            # Archives
            "*.zip",
            "*.tar",
            "*.gz",
            "*.tgz", 
            "*.rar",
            "*.7z",
            
            # Scripts (often utility, not core code)
            "*.bat",
            "*.sh",
            "*.ps1",
            
            # Binary files
            "*.bin",
            "*.exe",
            "*.dll",
            "*.so",
            "*.dylib",
            
            # Special files
            "VERSION",
            "AUTHORS",
            "LICENSE*",
            "COPYING*",
            "NOTICE*",
            "Makefile*",
            "Dockerfile*",
            "Procfile*",
            ".gitignore",
            ".gitattributes", 
            ".editorconfig",
            ".pylintrc",
            ".pycodestyle",
            ".pep8",
            ".bandit",
            ".coveragerc",
            ".flake8",
            ".dockerignore"
        },
        "javascript": {
            # Package files
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            
            # Documentation files
            "*.md",
            "*.mdx",
            "*.txt",
            "README*",
            "CHANGELOG*",
            
            # Configuration files
            "*.json",
            "*.config.js",
            "*.config.ts",
            ".eslintrc*",
            ".prettierrc*",
            "babel.config.*",
            "webpack.config.*",
            "rollup.config.*",
            "vite.config.*",
            "tsconfig*.json",
            
            # Environment files
            "*.env*",
            
            # Build files
            "Dockerfile*",
            "Makefile*",
            ".gitignore",
            ".gitattributes",
            ".editorconfig"
        },
        "java": {
            # Build files
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "gradle.properties",
            "settings.gradle",
            "*.sig",
            "*.exe",
            "*.default",
            "*.release",
            
            # Documentation
            "*.md",
            "*.txt",
            "README*",
            "CHANGELOG*",
            
            
            
            # Configuration
            "*.xml",
            "*.properties",
            "*.yml",
            "*.yaml",
            "*.json",
            
            # IDE files
            "*.iml",
            ".project",
            ".classpath",
            
            # Special files
            "LICENSE*",
            "NOTICE*",
            ".gitignore"
        },
        "c_and_cpp": {
            # Build System Files
            "Makefile*",
            "*.make",
            "*.ninja",
            "*.cmake",
            "CMakeCache.txt",
            "CMakeLists.txt",
            "*.vcxproj",
            "*.vcxproj.filters",
            "*.sln",
            "*.user",
            "*.suo",
            "configure",
            "config.h",
            "config.status",
            "config.log",
            "*.m4",
            "aclocal.m4",
            
            # Generated Files
            "*.o",
            "*.obj",
            "*.a",
            "*.lib",
            "*.so",
            "*.dll",
            "*.dylib",
            "*.exe",
            "*.out",
            "*.app",
            "*.pdb",
            "*.gch",
            "*.pch",
            
            # Documentation
            "*.md",
            "*.txt",
            "*.pdf",
            "README*",
            "CHANGELOG*",
            
            # Configuration
            "*.config",
            "*.ini",
            "*.json",
            "*.xml",
            "*.yaml",
            "*.yml",
            
            # Special files
            "LICENSE*",
            ".gitignore",
            ".clang-format",
            ".clang-tidy"
        },
        "rust": {
            # Build files
            "Cargo.toml",
            "Cargo.lock",
            
            # Documentation
            "*.md",
            "*.txt",
            "README*",
            "CHANGELOG*",
            
            # Configuration
            "*.toml",
            "*.json",
            "*.yaml",
            "*.yml",
            
            # Special files
            "LICENSE*",
            ".gitignore"
        },
        "go": {
            # Build files
            "go.mod",
            "go.sum",
            
            # Documentation
            "*.md",
            "*.txt",
            "README*",
            "CHANGELOG*",
            
            # Configuration
            "*.json",
            "*.yaml",
            "*.yml",
            "*.toml",
            
            # Special files
            "LICENSE*",
            ".gitignore",
            "Makefile*"
        }
    }
    
    def __init__(self, repo_slug: str, silent: bool = False, repo_cache: Optional[Union[str, Path]] = None, host: str = "github"):
        """
        Initialize the generator for multi-language repository analysis.
        
        Creates a new generator that will clone the specified repository and can extract
        function-level diffs from commits containing files in multiple languages.
        Parsers are loaded dynamically based on the languages detected in each commit.
        
        Args:
            repo_slug: Git repository identifier (e.g., 'owner/repo') to clone from hosting provider.
            silent: Whether to suppress progress bar
            repo_cache: Optional directory path to cache repositories persistently.
                       If provided, repos will be cached here instead of using temp dirs.
            host: Git hosting provider ('github' or 'gitlab'). Defaults to 'github'.
        """
        self.repo_path = None
        self.repo = None
        self.repo_slug = repo_slug
        self.host = host
        self.logger = logging.getLogger(__name__)
        self.silent = silent
        self.repo_cache = Path(repo_cache).expanduser() if repo_cache else None
        self.cleanup_repo = False  # Track whether to clean up on exit
        
        # Validate host
        if host not in self.HOST_URLS:
            raise ValueError(f"Unsupported host '{host}'. Supported hosts: {list(self.HOST_URLS.keys())}")
        
        # Dynamic parser and config storage
        self.parsers = {}  # language_name -> parser
        self.language_configs = {}  # language_name -> LanguageConfig
        
        # Clone repository using GitPython
        self.repo_path = self.clone_repository(repo_slug, host)
        self.logger.info(f"Repository available at {self.repo_path}")
        self.repo = Repo(self.repo_path)
    
    @classmethod 
    def create(cls, repo_slug: str, silent: bool = False, repo_cache: Optional[Union[str, Path]] = None, host: str = "github"):
        """
        Create a multi-language diff generator.
        
        Args:
            repo_slug: Repository identifier (e.g., 'owner/repo')
            silent: Whether to suppress progress bars
            repo_cache: Optional directory path to cache repositories persistently
            host: Git hosting provider ('github' or 'gitlab'). Defaults to 'github'.
            
        Returns:
            FuncLevelDiffGenerator that can handle multiple languages
        """
        return cls(repo_slug, silent, repo_cache, host)

    def _detect_file_language(self, file_path: str) -> Optional[str]:
        """
        Detect the programming language of a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language name if detected, None if unknown
        """
        _, ext = os.path.splitext(file_path.lower())
        
        # Map extensions to language names and their tree-sitter parser names
        extension_to_language = {
            # Python
            '.py': 'python',
            '.pyi': 'python', 
            '.pyx': 'python',
            '.pxi': 'python',
            
            # JavaScript/TypeScript
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',  # Use typescript parser for .ts files
            '.tsx': 'typescript',
            '.mjs': 'javascript',
            
            # Java
            '.java': 'java',
            
            # C/C++
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c++': 'cpp',
            '.h': 'c',  # Headers could be either, default to C
            '.hpp': 'cpp',
            '.hxx': 'cpp',
            '.h++': 'cpp',
            
            # C#
            '.cs': 'c_sharp',
            
            # Rust
            '.rs': 'rust',
            
            # Go
            '.go': 'go',
        }
        
        return extension_to_language.get(ext)

    def _get_language_config(self, language: str) -> LanguageConfig:
        """
        Get the language configuration for a given language.
        
        Args:
            language: Language name (e.g., 'python', 'javascript')
            
        Returns:
            LanguageConfig for the language
        """
        # Map parser language names to our config names
        language_mapping = {
            'python': 'python',
            'javascript': 'javascript', 
            'typescript': 'javascript',  # TypeScript uses JS config
            'java': 'java',
            'c': 'c_and_cpp',
            'cpp': 'c_and_cpp', 
            'c_sharp': 'csharp',
            'rust': 'rust',
            'go': 'go'
        }
        
        config_name = language_mapping.get(language)
        if config_name is None:
            return None
            
        # Get config from our predefined configs
        config_methods = {
            'python': LanguageConfig.python,
            'javascript': LanguageConfig.javascript,
            'java': LanguageConfig.java,
            'c_and_cpp': LanguageConfig.c_cpp,
            'csharp': LanguageConfig.csharp,
            'rust': LanguageConfig.rust,
            'go': LanguageConfig.go
        }
        
        return config_methods[config_name]()

    def _detect_languages_in_files(self, file_paths: List[str]) -> Set[str]:
        """
        Detect all programming languages present in a list of file paths.
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            Set of language names found in the files
        """
        languages = set()
        for file_path in file_paths:
            lang = self._detect_file_language(file_path)
            if lang:
                languages.add(lang)
        return languages

    def _load_parser(self, language: str):
        """
        Dynamically load a tree-sitter parser for the given language.
        
        Args:
            language: Language name (e.g., 'python', 'javascript')
        """
        if language in self.parsers:
            return  # Already loaded
            
        try:
            from tree_sitter_languages import get_parser
            parser = get_parser(language)
            self.parsers[language] = parser
            
            # Also store the language config
            config = self._get_language_config(language)
            if config:
                self.language_configs[language] = config
                
            self.logger.info(f"Loaded parser for {language}")
            
        except Exception as e:
            self.logger.warning(f"Could not load parser for {language}: {e}")

    def _load_parsers_for_commit(self, commit_hash: str):
        """
        Load all necessary parsers for files changed in the given commit.
        
        Args:
            commit_hash: The commit to analyze for language requirements
        """
        try:
            # Get all changed files in the commit
            changed_files = self.get_changed_files(commit_hash)
            
            # Detect languages in these files
            languages = self._detect_languages_in_files(changed_files)
            
            # Load parsers for each detected language
            for language in languages:
                self._load_parser(language)
                
        except Exception as e:
            self.logger.warning(f"Error loading parsers for commit {commit_hash}: {e}")

    def extract_functions_from_ast(self, text: str, parser, language_config: LanguageConfig) -> List[FunctionSpan]:
        """
        Extract function and method spans from source code using tree-sitter.
        
        Now takes parser and config as parameters to support multiple languages.
        
        Args:
            text: Source code text to parse
            parser: Tree-sitter parser instance for the appropriate language
            language_config: Language configuration for AST parsing
            
        Returns:
            List[FunctionSpan]: List of FunctionSpan objects representing all
                               functions and methods found in the code
        """
        tree = parser.parse(bytes(text, 'utf8'))
        functions = []
        
        def extract_from_node(node, class_name: Optional[str] = None):
            """Recursively extract functions from AST nodes."""
            # Check if this node is a function definition
            if node.type in language_config.function_node_types:
                func_name = self._find_identifier_in_node(node, text, language_config)
                
                if func_name:
                    # Convert byte offsets to line numbers
                    start_line = text[:node.start_byte].count('\n') + 1
                    end_line = text[:node.end_byte].count('\n') + 1
                    
                    functions.append(FunctionSpan(
                        name=func_name,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        start_line=start_line,
                        end_line=end_line,
                        class_name=class_name,
                        qualified_name_separator=language_config.qualified_name_separator
                    ))
            
            # Check if this node is a class definition
            elif node.type in language_config.class_node_types:
                class_name_for_methods = self._find_identifier_in_node(node, text, language_config)
                
                # Recursively process class body for methods
                for child in node.children:
                    extract_from_node(child, class_name_for_methods)
            
            # Recursively process child nodes
            else:
                for child in node.children:
                    extract_from_node(child, class_name)
        
        extract_from_node(tree.root_node)
        return functions

    def _find_identifier_in_node(self, node, text: str, language_config: LanguageConfig) -> Optional[str]:
        """
        Find identifier name in a node, handling language-specific patterns.
        
        Args:
            node: AST node to search
            text: Source code text
            language_config: Language configuration
            
        Returns:
            Identifier name if found, None otherwise
        """
        # Direct identifier child
        for child in node.children:
            if child.type == language_config.identifier_node_type:
                return text[child.start_byte:child.end_byte]
        
        # For some languages, identifier might be nested (e.g., in declarators)
        def find_identifier_recursive(n):
            if n.type == language_config.identifier_node_type:
                return text[n.start_byte:n.end_byte]
            for child in n.children:
                result = find_identifier_recursive(child)
                if result:
                    return result
            return None
        
        return find_identifier_recursive(node)

    def __enter__(self):
        """
        Context manager entry.
        
        Returns:
            self: The generator instance for use in the context
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - cleanup temporary directory.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any) 
            exc_tb: Exception traceback (if any)
        """
        self.cleanup()
    
    def cleanup(self):
        """
        Clean up temporary directory if it exists and cleanup is enabled.
        
        Only removes the cloned repository if it was created in a temporary directory.
        Cached repositories are preserved for future use.
        Called automatically when exiting context manager.
        """
        if self.cleanup_repo and self.repo_path and os.path.exists(self.repo_path):
            # For temp dirs, repo_path is like /tmp/function_diff_xyz/repo
            # We need to remove the parent temp directory
            temp_dir = os.path.dirname(self.repo_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            self.repo_path = None
            self.repo = None
    
    def clone_repository(self, repo_slug: str, host: str) -> str:
        """
        Clone a repository to either a cache directory or temporary directory using GitPython.
        
        If repo_cache is provided, attempts to use cached repository:
        - If repository exists in cache, pulls latest changes
        - If repository doesn't exist in cache, clones it there
        - Falls back to temporary directory if cache operations fail
        
        Args:
            repo_slug: Git repository identifier (e.g., 'owner/repo')
            host: Git hosting provider ('github' or 'gitlab')
            
        Returns:
            str: Path to the repository directory
            
        Raises:
            git.exc.GitError: If the repository cannot be cloned
        """
        # Build repository URL based on host
        base_url = self.HOST_URLS[host]
        repo_url = f"{base_url}/{repo_slug}.git"
        
        # Try to use cache if provided
        if self.repo_cache:
            try:
                return self._clone_to_cache(repo_url, repo_slug, host)
            except (OSError, PermissionError, git.exc.GitError) as e:
                self.logger.warning(f"Cache operation failed: {e}, falling back to temporary directory")
        
        # Fallback to temporary directory
        return self._clone_to_temp_dir(repo_url)

    def _get_cached_repo_path(self, repo_slug: str, host: str) -> Path:
        """Get the cache path for a repository."""
        owner, repo = repo_slug.split("/", 1)
        safe_repo_name = f"{host}--{owner}--{repo}"
        return self.repo_cache / safe_repo_name

    def _update_cached_repo(self, cached_repo_path: Path) -> bool:
        """
        Update an existing cached repository.
        
        Returns:
            bool: True if successfully updated, False if repo should be re-cloned
        """
        try:
            cached_repo = Repo(cached_repo_path)
            self.logger.info(f"Found cached repository at {cached_repo_path}, pulling latest changes")
            
            # Determine and update default branch
            try:
                origin = cached_repo.remotes.origin
                origin.fetch()
                
                # Try main first, then master
                default_branch = None
                if 'origin/main' in [ref.name for ref in cached_repo.refs]:
                    default_branch = 'main'
                elif 'origin/master' in [ref.name for ref in cached_repo.refs]:
                    default_branch = 'master'
                
                if default_branch:
                    cached_repo.git.checkout(default_branch)
                    origin.pull(default_branch)
                    self.logger.info(f"Updated cached repository from origin/{default_branch}")
                else:
                    self.logger.warning("Could not determine default branch, using repository as-is")
                    
            except git.exc.GitError as e:
                self.logger.warning(f"Could not pull latest changes: {e}, using cached repository as-is")
                return False
            
            return True
            
        except git.exc.InvalidGitRepositoryError:
            # Not a valid git repo, remove and re-clone
            self.logger.info(f"Invalid git repository in cache, removing {cached_repo_path}")
            shutil.rmtree(cached_repo_path, ignore_errors=True)
            return False

    def _clone_to_cache(self, repo_url: str, repo_slug: str, host: str) -> str:
        """Clone repository to cache directory."""
        # Create cache directory if it doesn't exist
        self.repo_cache.mkdir(parents=True, exist_ok=True)
        
        cached_repo_path = self._get_cached_repo_path(repo_slug, host)
        
        if cached_repo_path.exists():
            if self._update_cached_repo(cached_repo_path):
                self.cleanup_repo = False
                return str(cached_repo_path)
        
        # Clone into cache (either new repo or after failed update)
        self.logger.info(f"Cloning repository to cache at {cached_repo_path}")
        Repo.clone_from(repo_url, cached_repo_path)
        self.cleanup_repo = False
        return str(cached_repo_path)

    def _clone_to_temp_dir(self, repo_url: str) -> str:
        """Clone repository to temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix="function_diff_")
        repo_path = os.path.join(temp_dir, "repo")
        
        self.logger.info(f"Cloning repository to temporary directory at {repo_path}")
        Repo.clone_from(repo_url, repo_path)
        self.cleanup_repo = True
        return repo_path
    
    def get_file_at_commit(self, commit: str, file_path: str) -> str:
        """
        Get file contents at a specific commit using GitPython.
        
        Retrieves the contents of a file as it existed at a specific commit.
        Used to compare file states before and after changes.
        
        Args:
            commit: Git commit reference (e.g., 'HEAD~1', commit hash, branch name)
            file_path: Path to file within the repository (relative to repo root)
            
        Returns:
            str: File contents as string, or empty string if file doesn't exist
            
        Raises:
            git.exc.GitError: If the commit reference is invalid
            UnicodeDecodeError: If the file contains non-UTF-8 content
        """
        try:
            # Get the commit object
            commit_obj = self.repo.commit(commit)
            
            # Get the file blob at this commit
            blob = commit_obj.tree / file_path
            
            # Return the file contents as string
            return blob.data_stream.read().decode('utf-8')
            
        except (git.exc.GitError, KeyError, UnicodeDecodeError):
            # File might not exist at this commit or other Git errors
            return ""
    
    def get_diff_changed_lines(self, unified_diff: str) -> DiffRanges:
        """
        Extract line ranges that are changed in the unified diff.
        
        Parses the diff hunks to determine which line ranges were modified.
        Used to identify functions that overlap with changed regions.
        
        Args:
            unified_diff: The unified diff string to parse
            
        Returns:
            DiffRanges: Object containing:
                - source_ranges: line ranges in the pre-patch file 
                - target_ranges: line ranges in the post-patch file
        """
        try:
            patch_set = PatchSet(unified_diff)
            if not patch_set:
                return DiffRanges(source_ranges=[], target_ranges=[])
                
            patch = patch_set[0]
            source_ranges = []
            target_ranges = []
            
            for hunk in patch:
                # Handle different types of changes:
                # - For deletions: use source range (what was deleted)
                # - For additions: use target range (what was added)  
                # - For modifications: both ranges are similar, use both
                
                if hunk.source_length > 0:
                    # There's source content (deletions or modifications)
                    source_start = hunk.source_start
                    source_end = source_start + hunk.source_length - 1
                    source_ranges.append((source_start, source_end))
                
                if hunk.target_length > 0:
                    # There's target content (additions or modifications)
                    target_start = hunk.target_start  
                    target_end = target_start + hunk.target_length - 1
                    target_ranges.append((target_start, target_end))
                
            return DiffRanges(source_ranges=source_ranges, target_ranges=target_ranges)
            
        except Exception as e:
            raise e
            return DiffRanges(source_ranges=[], target_ranges=[])

    def get_a_and_b_paths(self, unified_diff: str) -> Tuple[str, str]:
        """
        Extract the a/ and b/ file paths from a unified diff header.
        
        Parses the unified diff to extract the file paths from the --- and +++ header lines.
        
        Args:
            unified_diff: The unified diff string to parse
            
        Returns:
            Tuple[str, str]: (a_path, b_path) where a_path is the source file path
                           and b_path is the target file path
        """
        try:
            patch_set = PatchSet(unified_diff)
            if not patch_set:
                return ("/dev/null", "/dev/null")
                
            patch = patch_set[0]
            
            # Get paths from the patch object
            a_path = patch.source_file or "/dev/null"
            b_path = patch.target_file or "/dev/null"
                
            return (a_path, b_path)
            
        except Exception as e:
            # If parsing fails, return default paths
            raise e
            return ("/dev/null", "/dev/null")

    def function_overlaps_changes(self, func_span: FunctionSpan, changed_ranges: List[Tuple[int, int]]) -> bool:
        """
        Check if a function's span overlaps with any of the changed line ranges.
        
        Determines whether a function definition contains or intersects with
        any of the lines that were modified in the diff.
        
        Args:
            func_span: The function span to check for overlap
            changed_ranges: List of (start_line, end_line) tuples for changed regions
            
        Returns:
            bool: True if the function overlaps with any changed range,
                 False if the function was not affected by changes
        """
        for start_line, end_line in changed_ranges:
            # Check for overlap: function_start <= change_end and function_end >= change_start
            if func_span.start_line <= end_line and func_span.end_line >= start_line:
                return True
        return False

    def generate_function_unified_diff(self, pre_text: str, post_text: str, a_path: str, b_path: str) -> str:
        """
        Generate a unified diff for a specific function.
        
        Creates a unified diff showing the changes made to a single function,
        with full context of the entire function definition.
        
        Args:
            pre_text: Function text before changes (from commit~1)
            post_text: Function text after changes (from commit)
            a_path: Source file path
            b_path: Target file path
        Returns:
            str: Unified diff string showing the changes within this function,
                with lines prefixed by -, +, or space for context
        """
        pre_lines = pre_text.splitlines(keepends=True)
        post_lines = post_text.splitlines(keepends=True)
            
        diff = difflib.unified_diff(
            pre_lines,
            post_lines,
            fromfile=a_path,
            tofile=b_path,
            n=max(len(pre_lines), len(post_lines))
        )
        
        return ''.join(diff)

    def __call__(self, commit_hash: str, max_history_scan_depth: int = 0) -> List[Dict[str, str]]:
        """
        Extract per-function unified diffs from a Git commit (main contract method).
        
        Args:
            commit_hash: The SHA of the commit that introduces the changes to analyze
            max_history_scan_depth: Maximum number of parent commits to traverse when searching for interesting commits
            
        Returns:
            List of function diff objects from all changed files
        """
        return self.extract_function_diffs_from_commit(commit_hash, max_history_scan_depth)

    def extract_function_diffs_from_file_diff(self, pre_patch_text: str, post_patch_text: str, 
                                             file_unified_diff: str, file_path: str) -> List[Dict[str, str]]:
        """
        Extract per-function unified diffs from a single file-level diff.
        
        Now detects the file's language and uses the appropriate parser and config.
        
        Args:
            pre_patch_text: The complete contents of the file before applying the patch
            post_patch_text: The complete contents of the file after applying the patch
            file_unified_diff: The unified diff for the file
            file_path: Path to the file (used for language detection)
            
        Returns:
            List of dictionaries with function diff information
        """
        # Detect file language
        file_language = self._detect_file_language(file_path)
        if not file_language:
            return []  # Unknown language, skip
            
        # Get parser and config for this language
        parser = self.parsers.get(file_language)
        language_config = self.language_configs.get(file_language)
        
        if not parser or not language_config:
            # Parser not available for this language
            return []
            
        # Parse both versions to get function spans
        pre_functions = self.extract_functions_from_ast(pre_patch_text, parser, language_config)
        post_functions = self.extract_functions_from_ast(post_patch_text, parser, language_config)
        
        # Get changed line ranges from the diff
        diff_ranges: DiffRanges = self.get_diff_changed_lines(file_unified_diff)
        
        # Get the a and b paths from the file_unified_diff
        a_path, b_path = self.get_a_and_b_paths(file_unified_diff)
        
        # Find functions that overlap with changes
        affected_functions = []
        
        # Create a mapping of function names to spans for post-patch
        post_func_map = {func.qualified_name: func for func in post_functions}
        
        for pre_func in pre_functions:
            # Compare pre-patch functions against source ranges
            if self.function_overlaps_changes(pre_func, diff_ranges.source_ranges):
                # Find corresponding function in post-patch
                post_func = post_func_map.get(pre_func.qualified_name)
                
                # Extract function text from both versions
                pre_func_text = pre_patch_text[pre_func.start_byte:pre_func.end_byte]
                
                if post_func:
                    post_func_text = post_patch_text[post_func.start_byte:post_func.end_byte]
                else:
                    # Function was deleted
                    post_func_text = ""
                
                # Generate per-function unified diff
                func_diff = self.generate_function_unified_diff(
                    pre_func_text,
                    post_func_text,
                    a_path,
                    b_path
                )
                
                if func_diff.strip():
                    affected_functions.append({
                        "func_name": pre_func.qualified_name,
                        "contextualized_changes": func_diff
                    })
        
        # Also check for newly added functions in post-patch
        pre_func_names = {func.qualified_name for func in pre_functions}
        
        for post_func in post_functions:
            if (post_func.qualified_name not in pre_func_names and 
                self.function_overlaps_changes(post_func, diff_ranges.target_ranges)):
                
                # This is a newly added function
                post_func_text = post_patch_text[post_func.start_byte:post_func.end_byte]
                
                func_diff = self.generate_function_unified_diff(
                    "",  # No pre-patch text for new function
                    post_func_text,
                    a_path,
                    b_path
                )
                
                if func_diff.strip():
                    affected_functions.append({
                        "func_name": post_func.qualified_name,
                        "contextualized_changes": func_diff
                    })
        
        return affected_functions

    def extract_function_diffs_from_commit(self, commit_hash: str, max_history_scan_depth: int = 0) -> List[Dict[str, str]]:
        """
        Extract per-function unified diffs for all files changed in the given commit.
        
        Now supports true multi-language commits by dynamically loading appropriate parsers.
        
        Args:
            commit_hash: The SHA of the commit that introduces the changes to analyze
            max_history_scan_depth: Maximum number of parent commits to traverse when searching for interesting commits
        Returns:
            List[Dict[str, str]]: List of function diff objects from all supported languages
                - func_name: The name of the function that was modified
                - contextualized_changes: The unified diff of the function
                - file_path: The path to the file that contains the function
                - file_language: The language of the file
        """
        all_function_diffs = []
        
        # Step 1: Find the first interesting commit
        if max_history_scan_depth:
            interesting_commit = self._get_interesting_commit(commit_hash, max_history_scan_depth)
        else:
            interesting_commit = commit_hash
        
        if interesting_commit is None:
            return []
        
        # Step 2: Load parsers for all languages in this commit
        self._load_parsers_for_commit(interesting_commit)
        
        # Step 3: Get all changed files with their diffs
        file_diffs = self.get_changed_files_with_diffs(interesting_commit)
        
        if not file_diffs:
            return []
        
        # Step 4: Process each changed file with its appropriate parser
        for file_path, file_diff_str in tqdm(file_diffs.items(), desc="Processing files", unit="files", disable=self.silent, leave=False):
            # Detect file language
            file_language = self._detect_file_language(file_path)
            if not file_language:
                continue  # Unknown file type
            
            # Check if we should ignore this file based on language-specific patterns
            language_for_ignore = self._get_language_config(file_language)
            if language_for_ignore and self._matches_ignore_patterns(file_path, language_for_ignore.name):
                continue
                
            # Check if we have a parser for this language
            if file_language not in self.parsers:
                continue  # No parser available
                
            try:
                # Get file contents for AST parsing
                pre_commit = f"{interesting_commit}~1"
                pre_patch_text = self.get_file_at_commit(pre_commit, file_path)
                post_patch_text = self.get_file_at_commit(interesting_commit, file_path)
                
                # Extract per-function diffs for this file
                if file_diff_str.strip():
                    function_level_diffs = self.extract_function_diffs_from_file_diff(
                        pre_patch_text, post_patch_text, file_diff_str, file_path
                    )
                    
                    # Add file path and language to each function diff
                    for diff in function_level_diffs:
                        diff.update({
                            "file_path": file_path,
                            "file_language": file_language
                        })
                    
                    all_function_diffs.extend(function_level_diffs)
                    
            except (git.exc.GitError, UnicodeDecodeError) as e:
                print(f"Warning: Could not process file {file_path}: {e}")
                continue
        
        if len(all_function_diffs) == 0:
            raise ValueError("No function diffs found")
        
        return all_function_diffs

    def get_changed_files(self, commit_hash: str) -> List[str]:
        """
        Get list of files changed in the given commit.
        
        Compares the commit with its parent to determine which files were
        added, modified, or deleted. This is the first step in the contract
        implementation.
        
        Args:
            commit_hash: The commit SHA to analyze (e.g., 'abc123', 'HEAD')
            
        Returns:
            List[str]: List of file paths that were changed in the commit,
                      relative to repository root
            
        Raises:
            git.exc.GitError: If the commit hash is invalid or not found
        """
        try:
            # Get the commit object
            commit_obj = self.repo.commit(commit_hash)
            
            # Get changed files by comparing with parent
            if commit_obj.parents:
                # Compare with first parent
                parent_commit = commit_obj.parents[0]
                diffs = parent_commit.diff(commit_obj, create_patch=False)
                
                changed_files = []
                for diff in diffs:
                    # Handle different types of changes
                    if diff.a_path:  # File was modified or deleted
                        changed_files.append(diff.a_path)
                    elif diff.b_path:  # File was added
                        changed_files.append(diff.b_path)
                
                return list(set(changed_files))  # Remove duplicates
            else:
                # Root commit - all files are "changed"
                return [item.path for item in commit_obj.tree.traverse() if item.type == 'blob']
                
        except git.exc.GitError as e:
            raise e

    def get_file_diff_from_commit(self, commit_hash: str, file_path: str) -> str:
        """
        Get the unified diff for a specific file from a Git commit using GitPython.
        
        Args:
            commit_hash: The commit SHA to analyze
            file_path: Path to the file within the repository
            
        Returns:
            str: Unified diff string for the file, or empty string if no changes
            
        Raises:
            git.exc.GitError: If the commit hash is invalid or not found
        """
        try:
            # Get the commit object
            commit_obj = self.repo.commit(commit_hash)
            
            if not commit_obj.parents:
                # Root commit - can't get diff
                return ""
            
            # Compare with first parent to get diffs
            parent_commit = commit_obj.parents[0]
            diffs = parent_commit.diff(commit_obj, paths=[file_path])
            
            if not diffs:
                return ""
            
            # Get the diff for our specific file
            for diff in diffs:
                if (diff.a_path == file_path or diff.b_path == file_path):
                    # Return the actual diff text
                    if diff.diff:
                        return diff.diff.decode('utf-8')
            
            return ""
            
        except (git.exc.GitError, UnicodeDecodeError) as e:
            raise e

    def get_changed_files_with_diffs(self, commit_hash: str) -> Dict[str, str]:
        """
        Get both the list of changed files and their unified diffs from a commit.
        
        Args:
            commit_hash: The commit SHA to analyze
            
        Returns:
            Dict[str, str]: Dictionary mapping file paths to their unified diff strings
            
        Raises:
            git.exc.GitError: If the commit hash is invalid or not found
        """
        try:
            # Get the commit object
            commit_obj = self.repo.commit(commit_hash)
            
            if not commit_obj.parents:
                # Root commit - all files are "added"
                return {}
            
            # Get diffs with patches
            parent_commit = commit_obj.parents[0]
            # List of diffs, one per file changed
            diffs = parent_commit.diff(commit_obj, create_patch=True)
            
            file_diffs = {}
            for diff in diffs:
                # Get the file path (prioritize b_path for new files, a_path for others)
                file_path = diff.b_path if diff.b_path else diff.a_path
                
                if file_path and diff.diff:
                    try:
                        diff_text = diff.diff.decode('utf-8')
                        a_path = f"--- a/{diff.a_path}" if diff.a_path else '--- /dev/null'
                        b_path = f"+++ b/{diff.b_path}" if diff.b_path else '+++ /dev/null'
                        
                        # Prepend diff headers
                        header = f"{a_path}\n{b_path}\n"
                        full_diff = header + diff_text

                        file_diffs[file_path] = full_diff
                    except UnicodeDecodeError:
                        # Skip binary files or files with encoding issues
                        continue
            
            return file_diffs
            
        except git.exc.GitError as e:
            raise e

    def _is_code_file(self, file_path: str, lang: str = "python") -> bool:
        """
        Check if a file is likely a source code file that should be processed.
        
        Uses both extension checking and ignore patterns to filter files.
        Filters files to only process source code files that can contain functions.
        Ignores documentation, configuration, test files, and binary files.
        
        Args:
            file_path: Path to the file (relative to repository root)
            lang: Programming language context for ignore patterns
            
        Returns:
            bool: True if the file appears to be a source code file that should
                 be processed for function extraction
        """
        # First check if the file matches ignore patterns
        if self._matches_ignore_patterns(file_path, lang):
            return False
        
        # Then check if it has a code extension
        code_extensions = {
            '.py', '.pxi', '.pyi', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.cxx', 
            '.h', '.hpp', '.hxx', '.cs', '.rs', '.go', '.php', '.rb', '.swift', 
            '.kt', '.scala', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.m', '.mm',
            '.r', '.R', '.jl', '.hs', '.elm', '.clj', '.cljs', '.fs', '.fsx', '.ml', '.mli'
        }
        
        _, ext = os.path.splitext(file_path.lower())
        return ext in code_extensions

    def _matches_ignore_patterns(self, filepath: str, lang: str = "python") -> bool:
        """
        Check if a file path matches any of the ignore patterns using gitignore-style matching.
        
        Determines whether a file should be ignored based on predefined patterns for
        documentation, configuration, test files, and other non-core code files.
        
        Args:
            filepath: Path to check (relative to repository root)
            lang: Programming language context (python, javascript, java, etc.)
            
        Returns:
            bool: True if the file should be ignored (not considered interesting code),
                 False if it should be processed
        """
        if lang not in self.ignore_patterns:
            return False
            
        # Normalize path separators
        normalized_path = filepath.replace('\\', '/')
        
        # Check directory patterns
        for pattern in self.ignore_patterns[lang]:
            if fnmatch.fnmatch(normalized_path, pattern):
                return True
            # Also check if any parent directory matches
            parts = normalized_path.split('/')
            for i in range(len(parts)):
                partial_path = '/'.join(parts[:i+1])
                if fnmatch.fnmatch(partial_path + '/', pattern):
                    return True
        
        # Check file patterns
        if lang in self.ignore_file_patterns:
            filename = os.path.basename(normalized_path)
            for pattern in self.ignore_file_patterns[lang]:
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(normalized_path, pattern):
                    return True
        
        return False

    def _is_interesting_commit(self, changed_files: List[str]) -> bool:
        """
        Check if a commit is interesting by examining all files with appropriate language patterns.
        """
        for filepath in changed_files:
            file_lang = self._detect_file_language(filepath)
            
            if file_lang is None:
                # Unknown file type, use python patterns as default
                if not self._matches_ignore_patterns(filepath, "python"):
                    return True
            else:
                # Use language-specific patterns  
                lang_config = self._get_language_config(file_lang)
                if lang_config and not self._matches_ignore_patterns(filepath, lang_config.name):
                    return True
        
        return False

    def _get_interesting_commit(self, commit_hash: str, max_history_scan_depth: int = 25) -> Optional[str]:
        """
        Find the first interesting commit (no longer tied to a single language).
        """
        try:
            current_commit = self.repo.commit(commit_hash)
            commits_checked = 0
            
            while commits_checked < max_history_scan_depth:
                changed_files = self.get_changed_files(current_commit.hexsha)
                
                if self._is_interesting_commit(changed_files):
                    return current_commit.hexsha
                
                commits_checked += 1
                
                if current_commit.parents:
                    current_commit = current_commit.parents[0]
                else:
                    break
            
            return None
            
        except git.exc.GitError as e:
            raise e
