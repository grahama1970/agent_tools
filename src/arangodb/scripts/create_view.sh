curl -X POST --user "root:openSesame" --header 'accept: application/json' --data-binary @- http://localhost:8529/_db/verifaix/_api/view <<EOF
{
  "name": "glossaryView",
  "type": "arangosearch",
  "links": {
    "glossary": {
      "includeAllFields": true,
      "fields": {
        "term": {
          "analyzers": ["text_en"]
        }
      }
    }
  }
}
EOF