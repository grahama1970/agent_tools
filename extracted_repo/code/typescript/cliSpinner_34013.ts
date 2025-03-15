import cliSpinners from 'cli-spinners';
import logUpdate from 'log-update';
import pc from 'picocolors';
import type { CliOptions } from './types.js';

export class Spinner {
  private spinner = cliSpinners.dots;
  private message: string;
  private currentFrame = 0;
  private interval: ReturnType<typeof setInterval> | null = null;
  private readonly isQuiet: boolean;

  constructor(message: string, cliOptions: CliOptions) {
    this.message = message;
    // If the user has specified the verbose flag, don't show the spinner
    this.isQuiet = cliOptions.quiet || cliOptions.verbose || false;
  }

  start(): void {
    if (this.isQuiet) {
      return;
    }

    const frames = this.spinner.frames;
    const framesLength = frames.length;
    this.interval = setInterval(() => {
      this.currentFrame++;
      const frame = frames[this.currentFrame % framesLength];
      logUpdate(`${pc.cyan(frame)} ${this.message}`);
    }, this.spinner.interval);
  }

  update(message: string): void {
    if (this.isQuiet) {
      return;
    }

    this.message = message;
  }

  stop(finalMessage: string): void {
    if (this.isQuiet) {
      return;
    }

    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    logUpdate(finalMessage);
    logUpdate.done();
  }

  succeed(message: string): void {
    if (this.isQuiet) {
      return;
    }

    this.stop(`${pc.green('✔')} ${message}`);
  }

  fail(message: string): void {
    if (this.isQuiet) {
      return;
    }

    this.stop(`${pc.red('✖')} ${message}`);
  }
}
