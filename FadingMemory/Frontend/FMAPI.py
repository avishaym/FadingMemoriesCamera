from flask import Flask, escape, request, render_template, jsonify
import logging
import sqlite3
from memories import generate_memory


def FMDB_getlastruls(howmany):
    conn = sqlite3.connect('/FadingMemory/Backend/FMDB/FMDB.db')
    c = conn.cursor()
    c.execute("SELECT url FROM images_metadata ORDER BY id DESC LIMIT :howmany", {'howmany':howmany})
    urllist = c.fetchmany(howmany)
    conn.close
    return urllist

def FMDB_registermail(url, email):
    conn = sqlite3.connect('/FadingMemory/Backend/FMDB/FMDB.db')
    c = conn.cursor()
    app.logger.info('%s %s about to register to FMDB', id, email)
    c.execute("UPDATE images_metadata SET email_addr=(:email) WHERE url=:url ", {'email':email, 'url':url})
    conn.commit()
    conn.close()

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def approval():
    if request.method == 'POST':
        # if not request.json or not 'title' in request.json:
        #     abort(400)
        approved = request.json['approved']
        idx, imagefile = generate_memory(approved)
        return jsonify(image_idx=idx,
                       image_url=imagefile)


@app.route('/gallery', methods=["GET", "POST"])
def gallery():
    if request.method == 'POST':
        if not request.json or not 'title' in request.json:
            abort(400)
        count = request.json['count']
        urllist = FMDB_getlastruls(count)
        print urllist
        return jsonify(urllist)

@app.route('/mailto', methods=["POST"])
def mailto():
    if request.method == 'POST':
        # if not request.json or not 'title' in request.json:
        #     abort(400)
        url = request.json['url']
        email = request.json['email']
        urllist = FMDB_registermail(url, email)
        print urllist
        return jsonify(urllist)

if __name__ == '__main__':
    print "in main"
    app.run(debug = True)
