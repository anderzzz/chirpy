'''Commands run to retrieve audio raw data

'''
from datetime import datetime
from rawdata import WebXenoCanto, RawDataHandler, web_xeno_canto_consts, web_xeno_canto_payload_consts

COL_SUBSET_BIG = ['catalogue_nr',
                  'generic_name', 'specific_name', 'subspecies_name', 'english_name',
                  'country_recorded', 'location_name',
                  'date_recording', 'time_of_day_recording',
                  'sound_type', 'license', 'quality_rating', 'length_recording']

def report_progress(say_what):
    print ('At {}: {}'.format(datetime.now(), say_what))

xeno = WebXenoCanto(web_xeno_canto_consts)
xeno.query(country='sweden',
           recording_quality_equal='A').get_all()
report_progress('Query string "{}" generated and executed'.format(xeno.query_string))
report_progress('Number of payload sets {}'.format(len(xeno)))

db = RawDataHandler(db_rootdir='./db_April26',
                    subfolder='audio',
                    payload_consts=web_xeno_canto_payload_consts)
for k_payload, payload in enumerate(xeno.payloads):
    report_progress('Downloading payload set {} / {}'.format(k_payload + 1, len(xeno)))

    db.populate_metadata(payload, col_subset=COL_SUBSET_BIG)
    db.download_audio(payload, sleep_seconds=1.0)

    report_progress('Current size of database: {}'.format(len(db)))