repomix -o custom-output.txt
repomix -i "*.log,tmp" -v
repomix -c ./custom-config.json
repomix --style xml
repomix --remote https://github.com/user/repo
npx repomix src