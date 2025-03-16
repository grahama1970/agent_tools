#!/usr/bin/env python3
"""
cli.py

Official Documentation:
- Click: https://click.palletsprojects.com/
- wget: https://www.gnu.org/software/wget/manual/wget.html
- tqdm: https://tqdm.github.io/

This CLI provides an interface for the fetch_docs project.
It allows an LLM agent (or user) to download documentation pages and process them into structured JSON chunks.
"""

import click
import json
import sys
import asyncio
from pathlib import Path

# Import functions from our project modules
from download_site import download_site
from main import process_directory

@click.group()
def cli():
    """
    Fetch Docs CLI

    A command-line tool for downloading documentation pages,
    cleaning and extracting their content, and producing an ordered JSON output
    with file and section hierarchies.
    """
    pass

@cli.command()
@click.argument('url')
@click.argument('output_dir', type=click.Path())
@click.option('--single-page', is_flag=True, help="Download only the provided page (non-recursive)")
def download(url, output_dir, single_page):
    """
    Download documentation pages from a given URL.

    URL: The root URL of the documentation (e.g., "https://docs.example.com/page")
    OUTPUT_DIR: The directory where downloaded pages will be stored.
    --single-page: If set, downloads only the single page provided (non-recursive).
    """
    click.echo(f"Starting download from {url} into directory: {output_dir} ...")
    try:
        download_site(url, output_dir, recursive=(not single_page))
        click.echo("Download completed successfully.")
    except Exception as e:
        click.echo(f"Error during download: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('site_dir', type=click.Path(exists=True))
@click.argument('output_json', type=click.Path())
def process(site_dir, output_json):
    """
    Process downloaded HTML pages into ordered JSON chunks.

    SITE_DIR: The directory containing the downloaded HTML files.
    OUTPUT_JSON: The path of the JSON file to output the processed data.
    """
    site_dir_path = Path(site_dir)
    click.echo(f"Processing HTML files in directory: {site_dir_path} ...")
    
    try:
        processed_data = process_directory(site_dir_path)
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        sys.exit(1)
    
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2)
        click.echo(f"Processing complete. JSON output written to: {output_json}")
    except Exception as e:
        click.echo(f"Error writing JSON output: {e}", err=True)
        sys.exit(1)

@cli.command()
def version():
    """
    Display the fetch_docs project version.
    """
    click.echo("fetch_docs version 1.0.0")

if __name__ == '__main__':
    cli()
