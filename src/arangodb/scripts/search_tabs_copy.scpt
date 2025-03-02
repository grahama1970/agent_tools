#!/usr/bin/osascript

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Quicklinks+
# @raycast.mode fullOutput
# @raycast.icon 🔍🔍
# @raycast.packageName Quicklinks+ 
# @raycast.argument1 { "type": "text", "placeholder": "URL/Keyword", "optional": false }

# Optional parameters:
# @raycast.description Find or create Chrome tab by URL/keyword match. Doesn't create a new tab if the tab is already open.
# @raycast.author graham_anderson
# @raycast.authorURL https://raycast.com/graham_anderson

on run argv
    set searchQuery to item 1 of argv
    set targetURL to getURLForQuery(searchQuery)

    tell application "Google Chrome"
        activate
        set foundTab to false
        
        -- Check all windows and tabs
        repeat with wIndex from 1 to number of windows
            set thisWindow to window wIndex
            repeat with tIndex from 1 to number of tabs in thisWindow
                set thisTab to tab tIndex of thisWindow
                set tabTitle to title of thisTab
                set tabURL to URL of thisTab
                
                if (tabTitle contains searchQuery) or (tabURL contains searchQuery) or (tabURL contains targetURL) then
                    set foundTab to true
                    set index of thisWindow to 1 -- Bring window to front
                    set active tab index of thisWindow to tIndex
                    exit repeat
                end if
            end repeat
            if foundTab then exit repeat
        end repeat

        if not foundTab then
            if (number of windows) = 0 then
                make new window
                set newWindow to front window
            else
                set newWindow to front window
            end if
            
            tell newWindow
                set newTab to make new tab with properties {URL: targetURL}
                set active tab index to (index of newTab)
            end tell
        end if
    end tell

    return ""
end run

on getURLForQuery(query)
    if query is "gh" then
        return "https://github.com"
    else if query is "g" then
        return "https://google.com"
    else if query is "rc" then
        return "https://raycast.com"
    else if query starts with "http" then
        return query
    else
        return "https://" & query
    end if
end getURLForQuery