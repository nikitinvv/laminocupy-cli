import sys
import time
import argparse
import os
from pathlib import Path
from datetime import datetime

from laminocupy_cli import logging
from laminocupy_cli import config
from laminocupy_cli import GPURecSteps

log = logging.getLogger(__name__)


def init(args):
    if not os.path.exists(str(args.config)):
        config.write(args.config)
    else:
        log.error("{0} already exists".format(args.config))    

def run_status(args):
    config.log_values(args)

def run_recstep(args):
    #config.show_config(args)
    file_name = Path(args.file_name)    
    if file_name.is_file():        
        t = time.time()
        clpthandle = GPURecSteps(args)        
        if(args.reconstruction_type=='full'):
            clpthandle.recon_steps()        
        if(args.reconstruction_type=='try'):            
            clpthandle.recon_steps_try()        
        log.warning(f'Reconstruction time {(time.time()-t):.01f}s')
    else:
        log.error("File Name does not exist: %s" % args.file_name)    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', **config.SECTIONS['general']['config'])    
    lamino_params = config.RECON_PARAMS
    lamino_steps_params = config.RECON_STEPS_PARAMS
    #
    
    cmd_parsers = [
        ('init',        init,            (),                                     "Create configuration file"),
        ('reconstep',   run_recstep,     lamino_steps_params,                    "Run laminographic reconstruction by splitting by chunks in z and angles (step-wise)"),
        ('status',      run_status,      lamino_steps_params,                    "Show the laminographic reconstruction status"),        
    ]

    subparsers = parser.add_subparsers(title="Commands", metavar='')

    for cmd, func, sections, text in cmd_parsers:
        cmd_params = config.Params(sections=sections)
        cmd_parser = subparsers.add_parser(cmd, help=text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cmd_parser = cmd_params.add_arguments(cmd_parser)
        cmd_parser.set_defaults(_func=func)

    args = config.parse_known_args(parser, subparser=True)
    # create logger
    logs_home = args.logs_home

    # make sure logs directory exists
    if not os.path.exists(logs_home):
        os.makedirs(logs_home)

    lfname = os.path.join(logs_home, 'laminocupyon_' + datetime.strftime(datetime.now(), "%Y-%m-%d_%H_%M_%S") + '.log')
    log_level = 'DEBUG' if args.verbose else "INFO"
    logging.setup_custom_logger(lfname, level=log_level)
    log.debug("Started laminocupyon")
    log.info("Saving log at %s" % lfname)

    try:
        args._func(args)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)




if __name__ == '__main__':
    main()
