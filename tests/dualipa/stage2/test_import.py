import pytest

def test_import_dualipa():
    try:
        from agent_tools.dualipa import __version__
        print(f"Successfully imported dualipa, version: {__version__}")
        assert __version__ is not None
    except ImportError as e:
        print(f"Import error: {e}")
        
        # Add debugging information
        import sys
        print(f"Python path: {sys.path}")
        
        import os
        try:
            import agent_tools
            print(f"agent_tools found at: {agent_tools.__file__}")
            agent_tools_dir = os.path.dirname(agent_tools.__file__)
            print(f"Contents of {agent_tools_dir}:")
            for item in os.listdir(agent_tools_dir):
                print(f"  - {item}")
        except ImportError:
            print("Could not import agent_tools at all")
            
        assert False, f"Failed to import dualipa: {e}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
