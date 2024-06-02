---
tags:
  - learning
  - tutorial
version 1.0: 2024-03-09T13:02:00
---
# Settings
## Links
Obsidian comes with two syntax of linking: wikilink and markdown link, and arguably right now it supports wikilink [better](https://help.obsidian.md/Linking+notes+and+files/Internal+links#Link+to+a+heading+in+a+note). For interoperability concerns, I use markdown link only. To do that, I 
1. Disable the `Use [[wikilinks]]` option in `Files and Links` setting to enable auto-generation of markdown links to the file when I drag that file to the note I am working on. 
2. Set `Relative path to file` in `new link format` to make sure it works outside of Obsidian in other markdown editors.

# Features
## Link to a block
*Not interoperable*. Use wikilink syntax `[[]]` and inside type `^` it will give you the "blocks" you can link to. Blocks are a unit of text (more granularity than headers). You can identify a block and a create a link to it by either: 
- no predefined block identifier: using the [search syntax](#Link%20Searching), once you select a block, obsidian will create a unique and random identifier and auto-append it to the end of the block, and auto-populate the identifier to the link that references the block. 
- you can create a human readable identifier by appending ` ^your_identifier` to the end of the block. Then you can reference it using `[[^your_identifier]]`. 

## Link searching 
Obsidian supports searching for files and headers to link to using wikilink syntax: 
- Type `[[]]`: 
	- without other things, it will show files to link to.
	- a `#` will show internal headers to link to
	- two `##`s will show headers to link to from across your vault.
	- `^` will show "blocks" you can link to. See [Link to a block](#Link%20to%20a%20block).
- Once you made the selection, it will automatically convert the link from the wikilink syntax, i.e. `[[]]`, to markdown syntax if you tweak the link settings shown [above](#Links). 

# Plugins
## [Obsidian git](https://github.com/denolehov/obsidian-git)
I use **iCloud** and **git** to sync my Obsidian vault: iCloud is used for continuous and real-time sync; git is used for snapshot syncing to which I can attach meaningful messages. More fine-grained file-level snapshot recording is achieved by adding date type file properties (`cmd + ;`). 

Obsidian git can automatically commit backups every x minutes, but since I already use iCloud and for each commit I want to customize the commit messages, I opt out for this feature. 
### Workflow: 
1. `git init` @ the local vault
2. use command palette (`cmd + p`) to `edit .gitignore`: add `.obsidian` to ignore obsidian files and `.DS_Store` to ignore OS files. 
3. create a remote private repo and add the remote repo. 
4. use `cmd + p` to find `Git: commit all changes with specific messages`(my hotkey: `cmd + m`). (bypassing the staging step)
5. use `Git: push` (my hotkey: `cmd + u`)to push the changes to remote. 

## [Full Calendar](https://davish.github.io/obsidian-full-calendar/)





# Resources
- [HackMD](https://hackmd.io/KSuvFGtVQcW8LR7G2d_-lA?view#Obsidian)
- [Dabi's 2hr tutorial](https://www.youtube.com/watch?v=WqKluXIra70)
## Note organization
### Maps of Content
- https://obsidian.rocks/quick-tip-quickly-organize-notes-in-obsidian/
- https://www.youtube.com/watch?v=AatZl1Z_n-g
	- css file that creates a dashboard-like homepage (good for navigation for myself and others if published)
### The Zettelkasten Method
- https://obsidian.rocks/getting-started-with-zettelkasten-in-obsidian/
- https://www.youtube.com/watch?v=Etr_Wyfpyvk
- https://medium.com/@fairylights_io/the-zettelkasten-method-examples-to-help-you-get-started-8f8a44fa9ae6
- https://www.youtube.com/watch?v=E6ySG7xYgjY
- https://www.youtube.com/watch?v=wvAZ9-hmWQU
### Para
- https://fortelabs.com/blog/para/
### LATCH
- [https://www.youtube.com/watch?v=vS-b_RUtL1A](https://www.youtube.com/watch?v=vS-b_RUtL1A)

## Zotoro
- [obsidian & zotoro](https://medium.com/@alexandraphelan/an-updated-academic-workflow-zotero-obsidian-cffef080addd)

## Plugins
- 