#!/bin/bash
# publish-sdk.sh — Publish SDKs to their respective registries.
# Usage: ./scripts/publish-sdk.sh <sdk-name> [version]
#
# Examples:
#   ./scripts/publish-sdk.sh typescript 0.1.0
#   ./scripts/publish-sdk.sh rust 0.1.0
#   ./scripts/publish-sdk.sh python 0.1.0

set -euo pipefail

SDK=$1
VERSION=${2:-"0.1.0"}

case $SDK in
  typescript)
    echo "Publishing magnitude-js@$VERSION to npm..."
    cd sdks/typescript
    npm version "$VERSION" --no-git-tag-version
    npm publish --access public
    echo "Done: npm install magnitude-js@$VERSION"
    ;;

  rust)
    echo "Publishing magnitude@$VERSION to crates.io..."
    cd sdks/rust
    # Update version in Cargo.toml
    sed -i "s/^version = .*/version = \"$VERSION\"/" Cargo.toml
    cargo publish
    echo "Done: cargo add magnitude@$VERSION"
    ;;

  python)
    echo "Publishing magnitude-client@$VERSION to PyPI..."
    cd python-client
    # Update version in pyproject.toml
    sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
    python -m build
    python -m twine upload dist/*
    echo "Done: pip install magnitude-client==$VERSION"
    ;;

  java)
    echo "Publishing com.magnitude:magnitude-java@$VERSION to Maven Central..."
    cd sdks/java
    # Update version in pom.xml
    sed -i "s/<version>.*</<version>$VERSION</" pom.xml
    mvn deploy
    echo "Done: <dependency><groupId>com.magnitude</groupId><artifactId>magnitude-java</artifactId><version>$VERSION</version></dependency>"
    ;;

  dotnet)
    echo "Publishing Magnitude.Client@$VERSION to NuGet..."
    cd sdks/dotnet
    dotnet pack -c Release -p:Version="$VERSION"
    dotnet push bin/Release/Magnitude.Client.*.nupkg -s https://api.nuget.org/v3/index.json
    echo "Done: dotnet add package Magnitude.Client --version $VERSION"
    ;;

  ruby)
    echo "Publishing magnitude@$VERSION to RubyGems..."
    cd sdks/ruby
    sed -i "s/s.version     = .*/s.version     = \"$VERSION\"/" magnitude.gemspec
    gem build magnitude.gemspec
    gem push magnitude-$VERSION.gem
    echo "Done: gem install magnitude -v $VERSION"
    ;;

  langchain-python)
    echo "Publishing langchain-magnitude@$VERSION to PyPI..."
    cd integrations/langchain-python
    sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
    python -m build
    python -m twine upload dist/*
    echo "Done: pip install langchain-magnitude==$VERSION"
    ;;

  llama-index)
    echo "Publishing llama-index-vector-stores-magnitude@$VERSION to PyPI..."
    cd integrations/llama-index
    sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
    python -m build
    python -m twine upload dist/*
    echo "Done: pip install llama-index-vector-stores-magnitude==$VERSION"
    ;;

  *)
    echo "Unknown SDK: $SDK"
    echo "Available: typescript, rust, python, java, dotnet, ruby, langchain-python, llama-index"
    exit 1
    ;;
esac
