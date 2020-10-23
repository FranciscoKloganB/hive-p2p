"""Generates new sphinx html docs, removes local user's paths from the generated
html, commits to the local repository and then updates docs at firebase website.
If no repository exists or no remote firebase website exists the commands fail.
"""
import os
import bs4

app_html_path = os.path.abspath(
    os.path.join(os.getcwd(), '..', 'docs', '_build', 'html', 'app.html'))

db = r"\\"


def remove_local_user_paths():
    token = f"{db}hive{db}"
    app_html_file = open(app_html_path, encoding="utf-8")
    soup = bs4.BeautifulSoup(app_html_file.read(), "html.parser")
    elements = soup.find_all("em", class_="property")
    for tag in elements:
        tag_text = tag.get_text()
        if token in tag_text:
            i = tag_text.find("'") + 1
            j = tag_text.find(token) + 2
            # exclude everything between the apostrophe and target token, discarding the first two backlashes
            tag_text = f"{tag_text[0:i]}{tag_text[j:]}".replace(db, "/")
            tag.string.replace_with(tag_text)
            print(tag)
    app_html_file.close()

    html = soup.prettify("utf-8")
    with open(app_html_path, "wb") as f:
        f.write(html)


if __name__ == "__main__":
    # generate docs
    # os.system("cd ..\\docs && make html")
    # fix documents
    remove_local_user_paths()
    # save on remote
    # os.system("git add -f * && git commit -m 'refresh docs' && git push")
    # # update website
    # os.system("firebase deploy")
