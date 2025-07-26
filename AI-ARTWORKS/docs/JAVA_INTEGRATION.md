# Java Integration in AI-ARTWORK

## Overview
AI-ARTWORK includes an embedded Oracle JDK (see `src/core/Oracle_JDK-24/`) to support features that require Java, such as certain image processing or bridge modules.

## Purpose
- Enables running Java-based components or plugins within the Python application.
- Facilitates integration with external tools or libraries that are only available in Java.

## Setup
- The JDK is bundled in `src/core/Oracle_JDK-24/` and should work out-of-the-box.
- No additional installation is required unless you wish to use a system JDK.

## Usage
- Java-dependent modules will automatically use the bundled JDK.
- If you encounter issues, ensure the JDK path is correct in your configuration or environment variables.

## Troubleshooting
- Check that the `JAVA_HOME` environment variable points to `src/core/Oracle_JDK-24/` if needed.
- For advanced configuration, see the official Oracle JDK documentation. 