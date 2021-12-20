# CHANGELOG

For now, the project relies on human-readable resources that sadly aren't static. That means the code sometimes has to change to keep up!

2021-12-20
----------

- make it easier to update the year of the last election
- handle new, separate pollster, client columns
- handle renaming of Brexit party to Reform party
- break out code for eliminating non-poll events from the table, update in light of new parsing behaviour
- handle new conventions in table (i.e. parse '?%' as NaN)
- handle new 'other' format
- remove calls to code for handling different pollster conventions on Nationalist parties (Plaid Cymru longer listed separately in table)
- change column name `other` to `others`
- update list of parties/style information
- update default plot range
- change some plotting defaults (newer version of matplotlib looks different)
